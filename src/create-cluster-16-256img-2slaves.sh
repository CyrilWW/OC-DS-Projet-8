#! /bin/bash
aws emr create-cluster --os-release-label 2.0.20221103.3 --applications Name=Hadoop Name=Spark --ec2-attributes '{"KeyName":"OC-DS-P8-EC2","InstanceProfile":"EMR_EC2_DefaultRole","SubnetId":"subnet-073d08dde410d041f","EmrManagedSlaveSecurityGroup":"sg-0fc9d982118c9a24d","EmrManagedMasterSecurityGroup":"sg-06473c7205d6acf80"}' --release-label emr-6.9.0 --log-uri 's3n://aws-logs-189404752973-eu-west-3/elasticmapreduce/' --steps '[{"Args":["spark-submit","--deploy-mode","cluster","s3://oc-ds-p8-fruits-project/fruits.py","16","2"],"Type":"CUSTOM_JAR","ActionOnFailure":"CONTINUE","Jar":"command-runner.jar","Properties":"","Name":"Application Spark"},{"Args":["spark-submit","--deploy-mode","cluster","s3://oc-ds-p8-fruits-project/fruits.py","32","2"],"Type":"CUSTOM_JAR","ActionOnFailure":"CONTINUE","Jar":"command-runner.jar","Properties":"","Name":"Application Spark"},{"Args":["spark-submit","--deploy-mode","cluster","s3://oc-ds-p8-fruits-project/fruits.py","64","2"],"Type":"CUSTOM_JAR","ActionOnFailure":"CONTINUE","Jar":"command-runner.jar","Properties":"","Name":"Application Spark"},{"Args":["spark-submit","--deploy-mode","cluster","s3://oc-ds-p8-fruits-project/fruits.py","128","2"],"Type":"CUSTOM_JAR","ActionOnFailure":"CONTINUE","Jar":"command-runner.jar","Properties":"","Name":"Application Spark"},{"Args":["spark-submit","--deploy-mode","cluster","s3://oc-ds-p8-fruits-project/fruits.py","256","2"],"Type":"CUSTOM_JAR","ActionOnFailure":"CONTINUE","Jar":"command-runner.jar","Properties":"","Name":"Application Spark"}]' --instance-groups '[{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":2}]},"InstanceGroupType":"MASTER","InstanceType":"c5.xlarge","Name":"Groupe d'\''instances maître - 1"},{"InstanceCount":2,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":2}]},"InstanceGroupType":"CORE","InstanceType":"c5.2xlarge","Name":"Groupe d'\''instances principal - 2"}]' --auto-terminate --auto-scaling-role EMR_AutoScaling_DefaultRole --bootstrap-actions '[{"Path":"s3://oc-ds-p8-fruits-project/bootstrap-emr.sh","Name":"Action personnalisée"}]' --ebs-root-volume-size 20 --service-role EMR_DefaultRole --auto-termination-policy '{"IdleTimeout":60}' --name 'WithSteps-16-256img-2slaves-2xlarge' --scale-down-behavior TERMINATE_AT_TASK_COMPLETION --region eu-west-3