from LIDS.data_utils import get_file_names, drop_meaningless_cols, drop_constant_features, min_max_scaler
from sklearn.preprocessing import LabelEncoder
from info_gain import info_gain
from sklearn.feature_selection import VarianceThreshold
import warnings
import numpy as np
import pandas as pd
import os
import pdb
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torch


def weighted_random_sampler(train):
    '''
    Args:
      train: train dataset (must have attribute `labels` as a 1D torch tensor of class ids)
    return:
      weighted_sampler: sampler to draw samples inversely proportional to class frequency.
    '''

    # Ensure we operate on CPU numpy ints
    targets_t = train.labels
    if isinstance(targets_t, torch.Tensor):
        targets_np = targets_t.detach().cpu().numpy().astype(int)
    else:
        targets_np = np.asarray(train.labels).astype(int)

    # Compute class weights using actual class ids (not assuming 0..K-1)
    classes, counts = np.unique(targets_np, return_counts=True)

    # Edge case: if only one class present, use uniform weights to avoid index errors
    if len(classes) == 1:
        samples_weight = np.ones_like(targets_np, dtype=np.float64)
    else:
        class_weight = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
        samples_weight = np.array([class_weight[c] for c in targets_np], dtype=np.float64)

    samples_weight_t = torch.from_numpy(samples_weight)
    weighted_sampler = WeightedRandomSampler(samples_weight_t, len(samples_weight_t))

    return weighted_sampler


def _prefix_for_choice(dataset_choice):
    if dataset_choice == 1:
        return 'CICDDoS2019'
    elif dataset_choice == 2:
        return 'BoT-IoT'
    elif dataset_choice == 3:
        return 'TON_IoT'
    return 'CICDDoS2019'


def _prefix_state_path():
    return os.path.join(os.getcwd(), 'Datasets', 'current_dataset_prefix.txt')


def set_current_prefix(prefix: str):
    try:
        base = os.path.join(os.getcwd(), 'Datasets')
        if not os.path.exists(base):
            os.makedirs(base)
        with open(_prefix_state_path(), 'w', encoding='utf-8') as f:
            f.write(prefix.strip())
    except Exception as e:
        print(f'Warning: failed to persist current dataset prefix: {e}')


def get_current_prefix(default: str = 'CICDDoS2019') -> str:
    try:
        with open(_prefix_state_path(), 'r', encoding='utf-8') as f:
            s = f.read().strip()
            return s if s else default
    except Exception:
        return default


def print_dataset_summary(df: pd.DataFrame, name: str = ''):
    try:
        print('Columns:', list(df.columns))
        n_rows, n_cols = df.shape
        num_cols = [c for c in df.columns if c != ' Label' and pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if c != ' Label' and not pd.api.types.is_numeric_dtype(df[c])]
        print(f"Dataset Summary{name and ' - ' + name}: rows={n_rows}, cols={n_cols}, numeric_features={len(num_cols)}, categorical_features={len(cat_cols)}")
        if ' Label' in df.columns:
            vc = df[' Label'].value_counts(dropna=False)
            print('Label distribution:', vc.to_dict())
            # class weights and average sample weight
            total = float(vc.sum()) if len(vc) else 0.0
            weights = {k: (0.0 if v == 0 else 1.0/float(v)) for k, v in vc.items()}
            avg_w = (sum(weights.get(lbl, 0.0) for lbl in df[' Label'])/n_rows) if n_rows else 0.0
            print('Class weights (1/count):', weights)
            print('Average sample weight:', round(avg_w, 6))
        # basic numeric means overview
        numeric_means = df[num_cols].mean().to_dict() if num_cols else {}
        if numeric_means:
            # Print only first few means to keep concise
            subset = dict(list(numeric_means.items())[:5])
            print('Feature means (sample):', subset, '...')
    except Exception as e:
        print(f'Warning: failed to print dataset summary: {e}')



def make_datasets(PATH, nrows, dataset_choice=1):
    """
    The function loads the processed dataset file and does
    1. inf records are deleted
    2. nan values deleted
    3. constant features removed
    4. duplicate features removed
    5. duplicates removed
    6. scaling
    7. less info gain features removed
    8. qasi constant features removed
    
    """

    # Load dataset based on user choice: 1=CICDDoS2019 (comma), 2=BoT-IoT (semicolon), 3=TON_IoT (comma)
    if dataset_choice == 2:
        dataset = load_dataset_botiot(PATH, nrows)
    elif dataset_choice == 3:
        dataset = load_dataset_toniot(PATH, nrows)
    else:
        dataset = load_dataset(PATH, nrows)

    dataset = dataset.replace([np.inf, -np.inf], np.nan)

    # Handle missing values more robustly: fill feature NaNs, only drop rows with missing labels
    # Identify label column
    lbl_col = ' Label' if ' Label' in dataset.columns else None
    for c in ['attack', 'label']:
        if lbl_col is None and c in dataset.columns:
            lbl_col = c
    if lbl_col is None:
        raise ValueError("Label column not found after loading. Expected one of [' Label','attack','label'].")

    # Standardize to ' Label' if not already
    if lbl_col != ' Label':
        dataset[' Label'] = dataset[lbl_col]
        if lbl_col in dataset.columns:
            dataset.drop(lbl_col, axis=1, inplace=True)
        lbl_col = ' Label'

    # Replace empty strings with NaN then fill feature NaNs with 0
    dataset.replace('', np.nan, inplace=True)
    feature_cols = [c for c in dataset.columns if c != lbl_col]
    if feature_cols:
        dataset[feature_cols] = dataset[feature_cols].fillna(0)

    # Drop rows with missing label only
    before_rows = len(dataset)
    dataset = dataset[dataset[lbl_col].notna()]
    dataset.reset_index(drop=True, inplace=True)
    print(f"After filling features and dropping missing labels: {dataset.shape} (dropped {before_rows - len(dataset)} rows)")

    dataset = drop_constant_features(dataset)
    dataset = drop_duplicate_features(dataset)
    print("Dataset Shape:", dataset.shape)
    dataset.drop_duplicates(inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    print("drop duplicates Dataset Shape:", dataset.shape)

    dataset = min_max_scaler(dataset)
    dataset = drop_less_information_gain_features(dataset)
    dataset = drop_qasi_constant_features(dataset)

    # Save processed dataset with dataset-specific prefix
    prefix = _prefix_for_choice(dataset_choice)
    set_current_prefix(prefix)
    out_path = os.path.join(os.getcwd(), 'Datasets', f'Processed_{prefix}.csv')
    dataset.to_csv(out_path)

    # print(mappings)
    print("Dataset Shape", dataset.shape)
    print("********************************* Dataset Created*****************************************")


def ensure_label_and_numeric(dataset, dataset_choice):
    # Standardize label column to ' Label' and convert features to numeric
    label_col = None
    if ' Label' in dataset.columns:
        label_col = ' Label'
        print(f"Using existing ' Label' column. Unique values: {dataset[' Label'].unique()}")
    elif 'attack' in dataset.columns:
        # For datasets with numeric attack column
        try:
            dataset[' Label'] = dataset['attack'].astype(int)
        except:
            dataset[' Label'] = dataset['attack']
        label_col = ' Label'
    elif 'label' in dataset.columns:
        # For BoT-IoT: label column contains strings like "DDoS - HTTP", keep as-is
        dataset[' Label'] = dataset['label']
        label_col = ' Label'
        print(f"Using 'label' column as ' Label'. Unique values: {dataset[' Label'].unique()}")
    elif 'type' in dataset.columns:
        # For TON_IoT, 'type' column should already be copied to ' Label'
        dataset[' Label'] = dataset['type']
        label_col = ' Label'
    elif 'Category' in dataset.columns:
        # Map any non-BENIGN to 1
        dataset[' Label'] = dataset['Category'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
        label_col = ' Label'
    else:
        raise ValueError('Unable to identify label column for the selected dataset.')

    # Drop original label-like columns other than the standardized one
    for c in ['attack', 'label', 'type', 'Category']:
        if c in dataset.columns and c != ' Label':
            dataset.drop(c, axis=1, inplace=True)

    # Convert non-numeric feature columns to numeric via label encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in list(dataset.columns):
        if col == ' Label':
            continue
        if dataset[col].dtype == 'object':
            # Normalize blanks to NaN for consistent handling
            series = dataset[col]
            series = series.replace('', np.nan)
            try:
                coerced = pd.to_numeric(series)
                dataset[col] = coerced
            except Exception:
                dataset[col] = le.fit_transform(series.astype(str).fillna('NA'))
    return dataset


def load_dataset_botiot(PATH, nrows):
    # Loop specific CSV files for BoT-IoT dataset with semicolon delimiter
    print("************************************Loading Files******************************************")
    print('Loading BoT-IoT dataset from folder: ', PATH)
    
    # Specify the BoT-IoT CSV files to load
    filenames = [
        'DDoS_HTTP.csv',
        'DDoS_TCP.csv',
        'DDoS_UDP.csv'
    ]
    
    print(f"Files to load: {filenames}")
    
    i = 0
    dataset = None
    for fname in filenames:
        print('*****************************************************************************************')
        print('*****************************************************************************************')
        print('File to Load: ', fname)
        
        fullp = os.path.join(PATH, fname)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                df = pd.read_csv(fullp, sep=';', quotechar='"', engine='python', on_bad_lines='skip', nrows=nrows)
            except TypeError:
                # Fallback for older pandas
                df = pd.read_csv(fullp, sep=';', quotechar='"', engine='python', error_bad_lines=False, nrows=nrows)
        
        print("***************** Original File Shape ******************")
        print(df.shape)
        
        # Merge category and subcategory columns for BoT-IoT
        if 'category' in df.columns and 'subcategory' in df.columns:
            print("Merging category and subcategory columns...")
            def merge_categories(row):
                cat = str(row['category']).strip()
                subcat = str(row['subcategory']).strip()
                # If both are Normal, return just "Normal"
                if cat.upper() == 'NORMAL' and subcat.upper() == 'NORMAL':
                    return 'Normal'
                # Otherwise concatenate with " - "
                return f"{cat} - {subcat}"
            
            df['label'] = df.apply(merge_categories, axis=1)
            print(f"Created 'label' column from category-subcategory merge")
            print(f"Unique labels: {df['label'].unique()}")
        
        df.dropna(axis=0, how='all', inplace=True)
        df.drop_duplicates(inplace=True)
        print("After null and duplicate:", df.shape)

        print("Column names before dropping duplicates: ", df.columns)
        
        if i == 0:
            dataset = df
        else:
            dataset = pd.concat([dataset, df], ignore_index=True)
            print("############ After Concatenation Dataset Shape ##############")
            print(dataset.shape)
            dataset.drop_duplicates(inplace=True)
            print('After duplicate Removal:', dataset.shape)
            del df
        
        i += 1
    
    print("***************After loading all files Dataset Shape***********:")
    print(dataset.shape)
    print("************************************ Files Loaded ******************************************")
    
    # Drop BoT-IoT specific columns
    drop_cols = ['record', 'flgs', 'proto', 'saddr', 'sport', 'dir', 'daddr', 'dport',
                 'state', 'srcid', 'soui', 'doui', 'sco', 'dco', 'attack', 'category', 'subcategory']
    to_drop = [c for c in drop_cols if c in dataset.columns]
    if to_drop:
        print(f'Dropping BoT-IoT columns: {to_drop}')
        dataset.drop(columns=to_drop, inplace=True)
    
    dataset = ensure_label_and_numeric(dataset, 2)
    print('BoT-IoT dataset prepared. Final Shape:', dataset.shape)
    return dataset


def load_dataset_toniot(PATH, nrows):
    # Loop specific CSV files for TON_IoT dataset with comma delimiter
    print("************************************Loading Files******************************************")
    print('Loading TON_IoT dataset from folder: ', PATH)
    
    # Specify the TON_IoT CSV files to load
    filenames = [
        'Network_dataset_1.csv',
        'Network_dataset_2.csv',
        'Network_dataset_3.csv',
        'Network_dataset_4.csv',
        'Network_dataset_5.csv',
        'Network_dataset_6.csv',
        'Network_dataset_7.csv',
        'Network_dataset_8.csv',
        'Network_dataset_9.csv',
        'Network_dataset_10.csv',
        'Network_dataset_11.csv',
        'Network_dataset_12.csv',
        'Network_dataset_13.csv',
        'Network_dataset_14.csv',
        'Network_dataset_15.csv',
        'Network_dataset_16.csv',
        'Network_dataset_17.csv',
        'Network_dataset_18.csv',
        'Network_dataset_19.csv',
        'Network_dataset_20.csv',
        'Network_dataset_21.csv',
        'Network_dataset_22.csv',
        'Network_dataset_23.csv'
    ]
    
    print(f"Files to load: {filenames} \n")
    
    i = 0
    dataset = None
    for fname in filenames:
        print('*****************************************************************************************')
        print('*****************************************************************************************')
        print('File to Load: ', fname)
        
        fullp = os.path.join(PATH, fname)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df = pd.read_csv(fullp, sep=',', engine='python', on_bad_lines='skip', nrows=nrows)
        
        print("*****************Original File Shape******************")
        print(df.shape)
        
        # Filter TON_IoT dataset to keep only 'normal' and 'ddos' types
        if 'type' in df.columns:
            print("Filtering TON_IoT dataset to keep only 'normal' and 'ddos' types...")
            before_filter = len(df)
            # Convert to lowercase for case-insensitive comparison
            df['type'] = df['type'].astype(str).str.lower().str.strip()
            
            # Drop rows with unwanted attack types
            # unwanted_types = ['injection', 'password', 'xss', 'ransomware', 'backdoor', 'scanning', 'dos', 'mitm']
            # df = df[~df['type'].isin(unwanted_types)]
            
            after_filter = len(df)
            dropped = before_filter - after_filter
            print(f"Filtered out {dropped} rows with unwanted attack types")
            print(f"Remaining types: {df['type'].unique()}")
        
        # For TON_IoT: Use 'type' column as ' Label' and drop 'label' column
        if 'type' in df.columns:
            print("Using 'type' column as ' Label' for TON_IoT dataset")
            df[' Label'] = df['type']
            print(f"Unique labels in 'type': {df[' Label'].unique()}")
            # Drop the 'label' column if it exists
            if 'label' in df.columns:
                df.drop('label', axis=1, inplace=True)
                print("Dropped 'label' column")
            # Drop the original 'type' column after copying to ' Label'
            if 'type' in df.columns:
                df.drop('type', axis=1, inplace=True)
        
        df.dropna(axis=0, how='all', inplace=True)
        df.drop_duplicates(inplace=True)
        print("After null and duplicate:", df.shape)
        
        if i == 0:
            dataset = df
        else:
            dataset = pd.concat([dataset, df], ignore_index=True)
            print("############After Concatenation Dataset Shape##############")
            print(dataset.shape)
            dataset.drop_duplicates(inplace=True)
            print('After duplicate Removal:', dataset.shape)
            del df
        
        i += 1
    
    print("***************After loading all files Dataset Shape***********:")
    print(dataset.shape)
    print("************************************ Files Loaded ******************************************")
    
    # Drop unwanted TON_IoT columns if present (including 'label' and 'ts')
    drop_cols = ['ts', 'src_ip','src_port','dst_ip','dst_port','proto','service','http_user_agent','http_orig_mime_types','http_resp_mime_types','weird_name','weird_addl','weird_notice','dns_AA','dns_RD','dns_RA','dns_rejected','ssl_version','ssl_cipher','ssl_resumed','ssl_established','ssl_subject','ssl_issuer','http_trans_depth','http_method','http_uri','http_referrer','http_version','dns_query', 'label']
    to_drop = [c for c in drop_cols if c in dataset.columns]
    if to_drop:
        print('Dropping TON_IoT columns:', to_drop)
        dataset.drop(columns=to_drop, inplace=True)
    
    dataset = ensure_label_and_numeric(dataset, 3)
    print_dataset_summary(dataset, name='TON_IoT (post-load)')
    print('TON_IoT dataset prepared. Final Shape:', dataset.shape)
    return dataset


def load_dataset(PATH, nrows):
    """
    1. load an individual file
    2. drop na and duplicates
    3. Concnate with the previous file that is loaded
    """

    # filenames= get_file_names(PATH)
    meaning_less_cols = ['Unnamed: 0', 'Flow ID', ' Timestamp', ' Source IP',
                         'SimillarHTTP', ' Source Port', ' Destination IP', ' Destination Port']

    filenames = ['DrDoS_SNMP.csv', 'DrDoS_DNS.csv', 'DrDoS_MSSQL.csv', 'DrDoS_SSDP.csv',
                 'DrDoS_NetBIOS.csv', 'DrDoS_LDAP.csv', 'DrDoS_NTP.csv', 'DrDoS_UDP.csv']

    print("************************************Loading Files******************************************")
    i = 0
    for item in filenames:
        print('*****************************************************************************************')
        print('*****************************************************************************************')

        print('File to Load: ', item)

        item = PATH + '/' + item
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if i == 0:

                dataset = pd.read_csv(item, nrows=nrows)
                print("*****************Original File Shape******************")
                print(dataset.shape)

                dataset = drop_meaningless_cols(dataset, meaning_less_cols)

                dataset.drop_duplicates(inplace=True)
                dataset.dropna(axis=0, inplace=True)
                print("After null and duplicate: ", dataset.shape)

            else:

                df1 = pd.read_csv(item, nrows=nrows)
                print("*****************Original File Shape***************")
                print(df1.shape)

                df1 = drop_meaningless_cols(df1, meaning_less_cols)
                df1.drop_duplicates(inplace=True)
                df1.dropna(axis=0, inplace=True)

                print("After null and duplicate:", df1.shape)

                dataset = pd.concat([dataset, df1])
                print("############After Concatenation Dataset Shape############## ")
                print(dataset.shape)
                dataset.drop_duplicates(inplace=True)

                print('After duplicate Removal:', dataset.shape)

                del df1

            i = i + 1

    print("***************After loading all files Dataset Shape***********:")
    print(dataset.shape)

    print("************************************ Files Loaded ******************************************")

    # dataset.to_csv('full_dataset.csv')

    return dataset


def drop_duplicate_features(dataset):
    print("*********************************Dropping Duplicate Features *****************************************")

    dup_features = []
    features = dataset.columns
    for i in range(len(features)):
        for j in range(i + 1, len(features)):

            x = dataset[features[i]] == dataset[features[j]]
            if x.sum() == len(x):
                dup_features.append(features[j])

    for i in range(len(dup_features)):
        print("{}. {}".format(i + 1, dup_features[i]))

    dataset.drop(dup_features, axis=1, inplace=True)

    print("*********************************Dropped Duplicate Features *****************************************")
    print("Dataset Shape: ", dataset.shape)
    return dataset


def drop_less_information_gain_features(dataset, threshold=0.05):
    '''
    Measures the reduction in entropy after the split  
    
    '''
    print("*******************************Deleting less info_gain features*************************")

    less_ig_cols = []

    features = list(set(dataset.columns) - set([' Label']))
    for col in features:
        info_gain_value = info_gain.info_gain(dataset[col], dataset[' Label'])
        if info_gain_value < threshold:
            less_ig_cols.append(col)
    i = 1
    for item in less_ig_cols:
        print("{}. {}".format(i, item))
        i = i + 1

    print("*******************************Less info_gain features Deleted*************************")

    dataset.drop(less_ig_cols, axis=1, inplace=True)
    print("Dataset Shape", dataset.shape)

    return dataset


def load_processed_dataset():
    print("********************************* Loading Preprocessed dataset********************************")
    prefix = get_current_prefix()
    PATH = os.path.join(os.getcwd(), 'Datasets', f'Processed_{prefix}.csv')
    if not os.path.exists(PATH):
        # Fallback to legacy name
        PATH = os.path.join(os.getcwd(), 'Datasets', 'Processed_Dataset.csv')
    dataset = pd.read_csv(PATH)
    if 'Unnamed: 0' in dataset.columns:
        dataset.drop('Unnamed: 0', axis=1, inplace=True)

    print(f"************************************ Preprocessed dataset Loaded ({os.path.basename(PATH)}) ****************************")

    return dataset


def load_pca_dataset():
    print("********************************* Loading PCA dataset********************************")

    base_path = os.path.join(os.getcwd(), 'Datasets')
    prefix = get_current_prefix()
    pca_path = os.path.join(base_path, f'PCA_{prefix}.csv')
    train_path = os.path.join(base_path, f'train_PCA_{prefix}.csv')
    test_path = os.path.join(base_path, f'test_PCA_{prefix}.csv')

    dataset = None
    if os.path.exists(pca_path):
        dataset = pd.read_csv(pca_path)
    else:
        parts = []
        if os.path.exists(train_path):
            parts.append(pd.read_csv(train_path))
        if os.path.exists(test_path):
            parts.append(pd.read_csv(test_path))
        if parts:
            dataset = pd.concat(parts, ignore_index=True)

    if dataset is None:
        raise FileNotFoundError("No PCA dataset found. Please run option 2 to create PCA datasets first.")

    # Drop index column if present
    if 'Unnamed: 0' in dataset.columns:
        dataset.drop('Unnamed: 0', axis=1, inplace=True)

    print("************************************ PCA dataset Loaded****************************")

    return dataset


def create_pca_dataset(components=41):
    print("****************************** Creating PCA Datasets *******************************")
    from sklearn.decomposition import PCA
    dataset = load_processed_dataset()
    encoder = LabelEncoder()
    dataset[' Label'] = encoder.fit_transform(dataset[' Label'])
    # Persist label encoder classes so we can map numeric labels back to strings later
    classes_path = os.path.join(os.getcwd(), 'Datasets', 'label_classes.npy')
    try:
        np.save(classes_path, encoder.classes_)
    except Exception as e:
        print(f"Warning: failed to save label classes to {classes_path}: {e}")
    
    dataset1 = dataset.copy()
    dataset1.drop(' Label', axis=1, inplace=True)

    # Ensure the number of PCA components does not exceed the available features
    n_features = dataset1.shape[1]
    if components is None or components <= 0:
        components = n_features
    if components > n_features:
        print(f"Requested PCA components ({components}) exceed available features ({n_features}). Reducing to {n_features}.")
        components = n_features

    pca = PCA(n_components=components)

    column_names = [f'PC {i + 1}' for i in range(components)]

    principal_components = pca.fit_transform(dataset1)

    principal_components = pd.DataFrame(principal_components, columns=column_names)

    train_PCA, test_PCA, train_label, test_label = train_test_split(principal_components, dataset[' Label'],
                                                                    test_size=0.33, random_state=42)

    train_PCA[' Label'] = train_label
    test_PCA[' Label'] = test_label

    print("****************************** PCA Datasets Created *******************************")
    prefix = get_current_prefix()
    out_dir = os.path.join(os.getcwd(), 'Datasets')
    # principal_components.to_csv(os.path.join(out_dir, f'PCA_{prefix}.csv'))  # optional full PCA
    train_PCA.to_csv(os.path.join(out_dir, f'train_PCA_{prefix}.csv'))
    test_PCA.to_csv(os.path.join(out_dir, f'test_PCA_{prefix}.csv'))
    print("****************************** Created PCA Datasets Saved *******************************")


def drop_qasi_constant_features(dataset):
    print("************************************ Dropping Qasi Constant Features ****************************")

    # Work on a copy without the label and only keep numeric columns
    df = dataset.copy()
    if ' Label' not in df.columns:
        print("Warning: ' Label' not found while dropping quasi-constant features; skipping step.")
        return dataset

    df_features = df.drop(columns=[' Label'])
    numeric_cols = [c for c in df_features.columns if pd.api.types.is_numeric_dtype(df_features[c])]
    if not numeric_cols:
        print("No numeric feature columns available for variance threshold. Skipping this step.")
        return dataset

    X = df_features[numeric_cols]
    # If X has zero columns or rows, skip safely
    if X.shape[1] == 0 or X.shape[0] == 0:
        print("Empty feature matrix encountered. Skipping variance threshold.")
        return dataset

    # Apply VarianceThreshold to identify low-variance features
    vt = VarianceThreshold(threshold=0.01)
    try:
        vt.fit(X)
    except Exception as e:
        print(f"VarianceThreshold.fit failed: {e}. Skipping this step.")
        return dataset

    support = vt.get_support()
    # True = keep, False = drop. We want to drop low-variance features (False)
    to_drop = [col for col, keep in zip(numeric_cols, support) if not keep]

    if to_drop:
        print("Dropping quasi-constant features (low variance):")
        for i, col in enumerate(to_drop, start=1):
            print(f"{i}. {col}")
        dataset.drop(columns=to_drop, inplace=True, errors='ignore')
    else:
        print("No quasi-constant features detected.")

    print("************************************ Dropped Qasi Constant Features ****************************")

    return dataset
