from aws_cdk import (
    Stack,
    aws_lambda as _lambda,
)
from constructs import Construct

class AwsLambdaSwipeatStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.build_lambda_func()

    def build_lambda_func(self):
        self.prediction_lambda = _lambda.DockerImageFunction(
            scope=self,
            id="DockerLambda",
            # Lambda Function name on AWSs
            function_name="DockerLambda",
            # build docker image when deploy
            code=_lambda.DockerImageCode.from_image_asset(
                directory="aws_lambda_swipeat/DockerLambda"
            ),
        )
