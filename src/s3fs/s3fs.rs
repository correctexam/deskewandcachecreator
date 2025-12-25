use s3::creds::Credentials;
use s3::error::S3Error;
use s3::request::ResponseData;
use s3::{Bucket, BucketConfiguration, Region};

pub fn get_object(
    server_url: &str,
    bucket_name: &str,
    s3_path: &str,
    login: &str,
    pass: &str,
) -> Result<ResponseData, S3Error> {
    // This requires a running minio server at localhost:9000

    let region = Region::Custom {
        region: "eu-central-1".to_owned(),
        endpoint:server_url.to_owned(),
    };
    //    let credentials = Credentials::default()?;
    let credentials = Credentials::new(Some(login), Some(pass), None, None, None)?;

    let mut bucket =
        Bucket::new(bucket_name, region.clone(), credentials.clone())?.with_path_style();

    if !(bucket.exists_blocking()?) {
        bucket = Bucket::create_with_path_style_blocking(
            bucket_name,
            region,
            credentials,
            BucketConfiguration::default(),
        )?
        .bucket;
    }

    let response_data: s3::request::ResponseData = bucket.get_object_blocking(s3_path)?;
    assert_eq!(response_data.status_code(), 200);
    Ok(response_data)
}


pub fn put_object(
    server_url: &str,
    bucket_name: &str,
    content: &[u8],
    s3_path: &str,
    login: &str,
    pass: &str,
) -> Result<ResponseData, S3Error> {
    // This requires a running minio server at localhost:9000

    let region = Region::Custom {
        region: "eu-central-1".to_owned(),
        endpoint: server_url.to_owned(),
    };
    //    let credentials = Credentials::default()?;
    let credentials = Credentials::new(Some(login), Some(pass), None, None, None)?;

    let mut bucket =
        Bucket::new(bucket_name, region.clone(), credentials.clone())?.with_path_style();

    if !(bucket.exists_blocking()?) {
        bucket = Bucket::create_with_path_style_blocking(
            bucket_name,
            region,
            credentials,
            BucketConfiguration::default(),
        )?
        .bucket;
    }

    let response_data: s3::request::ResponseData = bucket.put_object_blocking(s3_path,content)?;
    assert_eq!(response_data.status_code(), 200);
    Ok(response_data)
}
