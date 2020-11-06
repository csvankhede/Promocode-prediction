from pyspark.sql.types import *
from pyspark.sql.functions import *
from itertools import chain
import numpy as np
from pyspark.ml.feature import Imputer
from pyspark.sql import Window

# fill missing value using mean or median
def fill_missing(df,strategy='mean', missingValue=np.nan):
    """
    Fill missing value using statistical methods like mean and median
    
    parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
    strategy: str, default 'mean'
        strategy will be used to impute missing values
    missingValue: str,int,float,bool,np.nan, default np.nan
        The value will be considered as missing value
    
    returns 
    -------
    imp: pyspark.ml.feature.Imputer
    transformed df: pyspark.sql.dataframe.DataFrame
        missing value imputed dataframe
    """
    c_name = [n for n,d in df.dtypes if d != 'string' and d != 'boolean']
    imp = Imputer(inputCols=c_name,outputCols=c_name,strategy=strategy, missingValue=missingValue).fit(df)
    return imp,imp.transform(df)

# convert string to lower or upper case 
def case_convertion(df,l):
    """
    convert the str column to lower or upper case
    
    parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
    l: dict
        dictonary of the columns name and case to b converted, the value of case must be 'lower' or 'upper'. e.g. {'ST_NAME':'lower'}
    
    return
    ------
    pyspark.sql.dataframe.DataFrame
    """
    for i,case in l.items():
        if case=='lower':
            df = df.withColumn(i,lower(col(i).alias(i)))
        elif case=='upper':
            df = df.withColumn(i,upper(col(i).alias(i)))
    return df    

# replace value in column
def replace(df,d,col_name,keep=False):
    """
    replace value in dataframe column
    
    parameter
    ---------
    df: pyspark.sql.dataframe.DataFrame
    d: dictonary
        key value pair of the value and new value. e.g. {'Y':1,'N':0}
    col_name: str
        column name in the data frame in which replacement needs to be done
    keep: bool, default False
        if False the value not present in the dictonary will be replaced by null
    
    returns
    -------
    pyspark.sql.dataframe.DataFrame
    """
    mapping = create_map([lit(x) for x in chain(*d.items())])
    if keep:
        df = df.withColumn(col_name,coalesce(mapping[df[col_name]],df[col_name]).alias(col_name))
    else:
        df = df.withColumn(col_name,mapping[df[col_name]].alias(col_name))
    return df

# change col type
def change_col_type(df,schema):
    """
    Change the type of the column 
    
    parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
    schema: dict
        key value pair of the column name and type of the column, the type of the columns must be 'int','str','float' or 'bool'
        e.g. {'PID':'int','ST_NUM':'int','NUM_BEDROOMS':'int','NUM_BATH':'int','SQ_FT':'int','OWN_OCCUPIED':'int'}
        
    returns
    -------
    pyspark.sql.dataframe.DataFrame
    """
    d = {'int':IntegerType(),'str':StringType(),'float':FloatType(),'bool':BooleanType()}
    
    for c,t in schema.items():
        df = df.withColumn(c,col(c).cast(d[t]))
    return df

# find outlier
def detect_outlier(df,method='iqr',val=np.nan):
    """
    detect outlier in the dataframe and replace them
    
    parameter
    ---------
    df: pyspark.sql.dataframe.DataFrame
    method: str, default 'iqr'
        method for outlier detection, which include 'z_score','iqr' and 'std'
    val: str,int,float,bool,np.nan, default np.nan
        value to be replaced instead of the outlier
    
    returns
    -------
    pyspark.sql.dataframe.DataFrame
    """
    c_name = [n for n,d in df.dtypes if d != 'string' and d != 'boolean']
    if method=='z_score':
        for i in c_name:
            stat = df.select(mean(col(i)).alias('mean'),stddev(col(i)).alias('std')).collect()
            m = stat[0]['mean']
            s = stat[0]['std']
            df =  df.withColumn(i,when(abs((col(i)-m)/s)>thresh,val).otherwise(col(i)))
    elif method=='iqr':
        for i in c_name:
            q1,q3 = df.approxQuantile(i,[0.25,0.75],0)
            IQR = q3-q1
            lo = q1-(1.5*IQR)
            up = q3+(1.5*IQR)
            df = df.withColumn(i,when(col(i).between(lo,up), col(i)).otherwise(val))
    elif method=='std':
        for i in c_name:
            stat = df.select(mean(col(i)).alias('mean'),stddev(col(i)).alias('std')).collect()
            m = stat[0]['mean']
            s = stat[0]['std']*thresh
            lo = m - s
            up = m + s
            df = df.withColumn(i,when(col(i).between(lo,up), col(i)).otherwise(val))
    return df

# split column by regular expression
def split_column(df,col_name,reg_ex=',',keep=False):
    """
    split single column into multiple columns using regular expression
    
    parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
    col_name: str
        name of the column to be split in dataframe
    reg_ex: str
        regular expression for spliting
    keep: bool, default False
        remove original column if set to False
    
    returns
    -------
    pyspark.sql.dataframe.DataFrame
    """
    # https://stackoverflow.com/a/51680292/5847441
    df = df.select(col_name,posexplode(split(col_name,reg_ex)).alias('pos','val'))\
        .select(col_name,concat(lit(col_name),col('pos').cast('string')).alias('name'),'val')\
        .groupBy(col_name).pivot('name').agg(first('val'))
    if keep:
        return df
    else:
        return df.drop(col_name)

# trim space
def trim_space(df,col_name):
    """
    trim extra space from starting and ending in the column
    
    parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
    col_name: str
        column name in the dataframe
    
    returns
    -------
    pyspark.sql.dataframe.DataFrame
    """
    return df.withColumn(col_name,trim(col_name))


# count of unique value in column
def get_counts(df,col_name):
    """
    Give count of each unique value in the column of the dataframe
    
    parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
    col_name: str
        column name in the dataframe
    
    returns
    -------
    pyspark.sql.dataframe.DataFrame
    """
    return df.groupBy(col_name).count().show()

# remove column
def remove_column(df,col_name):
    """
    Remove columns from dataframe
    
    parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
    col_name: str
        column name in the dataframe
    
    returns
    -------
    pyspark.sql.dataframe.DataFrame
    """
    return df.drop(col_name)

# drop duplicates
def drop_dups(df,col_names=None):
    """
    drop duplicate rows from the dataframe
    
    parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
    col_name: str
        column name in the dataframe
    
    returns
    -------
    pyspark.sql.dataframe.DataFrame
    """
    return df.dropDuplicates()

# insert column 
def add_column(df,col_name,use_func=False,func=None,data=None):
    """
    Add new colunm to the dataframe
    
    parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
    col_name: str
        column name in the dataframe
    use_func: bool, default False
        if set to True it will add new column using the function provided in func argument
        else the list or np.ndarray will be added in the dataframe which user have to provide in data argument
    func: function
        It will be used with use_func argument, e.g. exp('SQ_FT'), log('SQ_FT')
    data: list, np.ndarray
        data to be filled in new column
    
    returns
    -------
    pyspark.sql.dataframe.DataFrame
    """
    if use_func:
        df = df.withColumn(col_name,func)
    else:
        if type(data)==np.ndarray:
            data = data.tolist()
        a = spark.createDataFrame([(i,) for i in data],[col_name])
        a = a.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
        df = df.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
        df = df.join(a,df.row_idx==a.row_idx).drop('row_idx')
    return df