# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the KubernetesBackend class in the Kubeflow Optimizer SDK.

This module uses pytest and unittest.mock to simulate Kubernetes API interactions.
It tests KubernetesBackend's behavior across job listing, optimization, deletion etc.
"""

from dataclasses import asdict
import datetime
import multiprocessing
from unittest.mock import Mock, patch

from kubeflow_katib_api import models
import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.optimizer.backends.kubernetes.backend import KubernetesBackend
from kubeflow.optimizer.constants import constants
from kubeflow.optimizer.types.algorithm_types import RandomSearch
from kubeflow.optimizer.types.optimization_types import (
    Direction,
    Metric,
    Objective,
    OptimizationJob,
    Result,
    TrialConfig,
)
from kubeflow.optimizer.types.search_types import (
    ContinuousSearchSpace,
    Distribution,
    Search,
)
from kubeflow.trainer.constants import constants as trainer_constants
from kubeflow.trainer.test.common import (
    DEFAULT_NAMESPACE,
    FAILED,
    RUNTIME,
    SUCCESS,
    TIMEOUT,
    TestCase,
)
from kubeflow.trainer.types.types import (
    CustomTrainer,
    Event,
    TrainJob,
    TrainJobTemplate,
)

BASIC_OPTIMIZATION_JOB_NAME = "test-optimization-job"
BASIC_TRIAL_NAME = "trial-abc123"
CREATION_TIMESTAMP = datetime.datetime(2025, 6, 1, 10, 30, 0)


# --------------------------
# Fixtures
# --------------------------


@pytest.fixture
def kubernetes_backend():
    """Provide a KubernetesBackend with mocked Kubernetes APIs."""

    mock_trainjob = Mock(spec=TrainJob)
    mock_trainjob.name = BASIC_TRIAL_NAME
    mock_trainjob.status = trainer_constants.TRAINJOB_COMPLETE
    mock_step = Mock()
    mock_step.name = trainer_constants.NODE + "-0"
    mock_step.status = "Running"
    mock_step.pod_name = "trial-pod-0"
    mock_trainjob.steps = [mock_step]

    with (
        patch("kubernetes.config.load_kube_config", return_value=None),
        patch(
            "kubernetes.client.CustomObjectsApi",
            return_value=Mock(
                create_namespaced_custom_object=Mock(side_effect=conditional_error_handler),
                get_namespaced_custom_object=Mock(
                    side_effect=get_namespaced_custom_object_response
                ),
                list_namespaced_custom_object=Mock(
                    side_effect=list_namespaced_custom_object_response
                ),
                delete_namespaced_custom_object=Mock(side_effect=conditional_error_handler),
            ),
        ),
        patch(
            "kubernetes.client.CoreV1Api",
            return_value=Mock(
                list_namespaced_event=Mock(side_effect=mock_list_namespaced_event),
                read_namespaced_pod_log=Mock(side_effect=mock_read_namespaced_pod_log),
                list_namespaced_pod=Mock(return_value=Mock(get=Mock(return_value=Mock(items=[])))),
            ),
        ),
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.verify_backend",
            return_value=None,
        ),
    ):
        backend = KubernetesBackend(KubernetesBackendConfig())
        backend.trainer_backend.get_job = Mock(return_value=mock_trainjob)
        backend.trainer_backend._get_trainjob_spec = Mock(
            return_value=Mock(to_dict=Mock(return_value={"mock": "spec"}))
        )
        backend.trainer_backend._read_pod_logs = Mock(return_value=iter(["test log content"]))
        yield backend


@pytest.fixture
def optimizer_backend():
    """Provide an optimizer KubernetesBackend with mocked Kubernetes APIs."""
    with (
        patch("kubernetes.config.load_kube_config", return_value=None),
        patch(
            "kubernetes.client.CustomObjectsApi",
            return_value=Mock(
                create_namespaced_custom_object=Mock(return_value=None),
            ),
        ),
        patch("kubernetes.client.CoreV1Api", return_value=Mock()),
        patch(
            "kubeflow.trainer.backends.kubernetes.backend.KubernetesBackend.verify_backend",
            return_value=None,
        ),
    ):
        backend = KubernetesBackend(KubernetesBackendConfig())
        backend.trainer_backend._get_trainjob_spec = Mock(
            return_value=Mock(to_dict=Mock(return_value={}))
        )
        yield backend


# --------------------------
# Mock Handlers
# --------------------------


def conditional_error_handler(*args, **kwargs):
    """Raise simulated errors based on resource name."""
    if args[2] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    elif args[2] == RUNTIME:
        raise RuntimeError()


def get_namespaced_custom_object_response(*args, **kwargs):
    """Return a mocked Experiment CR via mock async thread."""
    mock_thread = Mock()
    if args[2] == TIMEOUT or args[4] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    if args[2] == RUNTIME or args[4] == RUNTIME:
        raise RuntimeError()

    mock_thread.get.return_value = get_experiment(name=args[4]).to_dict()
    return mock_thread


def list_namespaced_custom_object_response(*args, **kwargs):
    """Return ExperimentList or TrialList via mock async thread."""
    mock_thread = Mock()
    if args[2] == TIMEOUT:
        raise multiprocessing.TimeoutError()
    if args[2] == RUNTIME:
        raise RuntimeError()

    plural = args[3]

    if plural == constants.EXPERIMENT_PLURAL:
        mock_thread.get.return_value = get_experiment_list_response()
    elif plural == constants.TRIAL_PLURAL:
        mock_thread.get.return_value = get_trial_list_response()

    return mock_thread


def mock_list_namespaced_event(*args, **kwargs):
    """Simulate event listing from namespace."""
    mock_thread = Mock()
    mock_thread.get.return_value = models.IoK8sApiCoreV1EventList(
        items=[
            models.IoK8sApiCoreV1Event(
                metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                    name="experiment-event-1",
                    namespace=DEFAULT_NAMESPACE,
                ),
                involvedObject=models.IoK8sApiCoreV1ObjectReference(
                    kind=constants.EXPERIMENT_KIND,
                    name=BASIC_OPTIMIZATION_JOB_NAME,
                    namespace=DEFAULT_NAMESPACE,
                ),
                message="Experiment created successfully",
                reason="Created",
                firstTimestamp=datetime.datetime(2025, 6, 1, 10, 30, 0),
            ),
            models.IoK8sApiCoreV1Event(
                metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                    name="trial-event-1",
                    namespace=DEFAULT_NAMESPACE,
                ),
                involvedObject=models.IoK8sApiCoreV1ObjectReference(
                    kind=constants.TRIAL_KIND,
                    name=BASIC_TRIAL_NAME,
                    namespace=DEFAULT_NAMESPACE,
                ),
                message="Trial started",
                reason="Running",
                firstTimestamp=datetime.datetime(2025, 6, 1, 10, 31, 0),
            ),
            # Unrelated event that should be filtered out.
            models.IoK8sApiCoreV1Event(
                metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                    name="unrelated-event",
                    namespace=DEFAULT_NAMESPACE,
                ),
                involvedObject=models.IoK8sApiCoreV1ObjectReference(
                    kind="Pod",
                    name="some-pod",
                    namespace=DEFAULT_NAMESPACE,
                ),
                message="Pod scheduled",
                reason="Scheduled",
                firstTimestamp=datetime.datetime(2025, 6, 1, 10, 32, 0),
            ),
        ]
    )
    return mock_thread


def mock_read_namespaced_pod_log(*args, **kwargs):
    """Simulate log retrieval from a pod."""
    return "test log content"


# --------------------------
# Object Creators
# --------------------------


def get_experiment(
    name: str = BASIC_OPTIMIZATION_JOB_NAME,
) -> models.V1beta1Experiment:
    """Create a mock Experiment object."""
    return models.V1beta1Experiment(
        apiVersion=constants.API_VERSION,
        kind=constants.EXPERIMENT_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=name,
            namespace=DEFAULT_NAMESPACE,
            creationTimestamp=CREATION_TIMESTAMP,
        ),
        spec=models.V1beta1ExperimentSpec(
            parameters=[
                models.V1beta1ParameterSpec(
                    name="learning_rate",
                    parameterType=constants.DOUBLE_PARAMETER,
                    feasibleSpace=models.V1beta1FeasibleSpace(
                        min="0.001",
                        max="0.1",
                        distribution="uniform",
                    ),
                ),
            ],
            objective=models.V1beta1ObjectiveSpec(
                objectiveMetricName="loss",
                type="minimize",
            ),
            algorithm=models.V1beta1AlgorithmSpec(
                algorithmName="random",
            ),
            maxTrialCount=10,
            parallelTrialCount=1,
        ),
        status=models.V1beta1ExperimentStatus(
            conditions=[
                models.V1beta1ExperimentCondition(
                    type=constants.EXPERIMENT_SUCCEEDED,
                    status="True",
                )
            ],
            currentOptimalTrial=models.V1beta1OptimalTrial(
                bestTrialName=BASIC_TRIAL_NAME,
                parameterAssignments=[
                    models.V1beta1ParameterAssignment(
                        name="learning_rate",
                        value="0.01",
                    )
                ],
                observation=models.V1beta1Observation(
                    metrics=[
                        models.V1beta1Metric(
                            name="loss",
                            latest="0.05",
                            max="0.1",
                            min="0.02",
                        )
                    ]
                ),
            ),
        ),
    )


def get_experiment_list_response() -> dict:
    """Return dict for ExperimentList."""
    experiment_list = models.V1beta1ExperimentList(
        items=[
            get_experiment(name="opt-job-1"),
            get_experiment(name="opt-job-2"),
        ]
    )
    return experiment_list.to_dict()


def get_trial_list_response() -> dict:
    """Return dict for TrialList."""
    trial_list = models.V1beta1TrialList(
        items=[
            models.V1beta1Trial(
                metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                    name=BASIC_TRIAL_NAME,
                    namespace=DEFAULT_NAMESPACE,
                ),
                spec=models.V1beta1TrialSpec(
                    parameterAssignments=[
                        models.V1beta1ParameterAssignment(
                            name="learning_rate",
                            value="0.01",
                        )
                    ],
                ),
                status=models.V1beta1TrialStatus(
                    observation=models.V1beta1Observation(
                        metrics=[
                            models.V1beta1Metric(
                                name="loss",
                                latest="0.05",
                                max="0.1",
                                min="0.02",
                            )
                        ]
                    ),
                ),
            ),
        ]
    )
    return trial_list.to_dict()


# --------------------------
# Tests
# --------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with defaults",
            expected_status=SUCCESS,
            config={
                "search_space": {"learning_rate": Search.uniform(0.001, 0.1)},
            },
        ),
        TestCase(
            name="valid flow with custom objectives and algorithm",
            expected_status=SUCCESS,
            config={
                "search_space": {"learning_rate": Search.uniform(0.001, 0.1)},
                "objectives": [
                    Objective(metric="accuracy", direction=Direction.MAXIMIZE),
                ],
                "algorithm": RandomSearch(random_state=42),
            },
        ),
        TestCase(
            name="empty search space raises ValueError",
            expected_status=FAILED,
            config={
                "search_space": {},
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="timeout error when creating experiment",
            expected_status=FAILED,
            config={
                "namespace": TIMEOUT,
                "search_space": {"learning_rate": Search.uniform(0.001, 0.1)},
            },
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when creating experiment",
            expected_status=FAILED,
            config={
                "namespace": RUNTIME,
                "search_space": {"learning_rate": Search.uniform(0.001, 0.1)},
            },
            expected_error=RuntimeError,
        ),
    ],
)
def test_optimize(kubernetes_backend, test_case):
    """Test KubernetesBackend.optimize and Experiment CR creation."""
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)

        trial_template = TrainJobTemplate(
            trainer=Mock(func_args=None),
        )

        name = kubernetes_backend.optimize(
            trial_template=trial_template,
            search_space=test_case.config["search_space"],
            objectives=test_case.config.get("objectives"),
            algorithm=test_case.config.get("algorithm"),
        )

        assert test_case.expected_status == SUCCESS
        assert isinstance(name, str)
        assert len(name) == 12

        kubernetes_backend.custom_api.create_namespaced_custom_object.assert_called_once()
        call_args = kubernetes_backend.custom_api.create_namespaced_custom_object.call_args
        assert call_args[0][0] == constants.GROUP
        assert call_args[0][1] == constants.VERSION
        assert call_args[0][2] == DEFAULT_NAMESPACE
        assert call_args[0][3] == constants.EXPERIMENT_PLURAL

        experiment_dict = call_args[0][4]
        assert experiment_dict["apiVersion"] == constants.API_VERSION
        assert experiment_dict["kind"] == constants.EXPERIMENT_KIND
        assert experiment_dict["metadata"]["name"] == name

        spec = experiment_dict["spec"]
        param_names = [p["name"] for p in spec["parameters"]]
        assert param_names == list(test_case.config["search_space"].keys())

        expected_objectives = test_case.config.get("objectives", [Objective()])
        assert spec["objective"]["objectiveMetricName"] == expected_objectives[0].metric
        assert spec["objective"]["type"] == expected_objectives[0].direction.value

        expected_algorithm = test_case.config.get("algorithm", RandomSearch())
        assert spec["algorithm"]["algorithmName"] == expected_algorithm.algorithm_name

        assert spec["trialTemplate"]["retain"] is True
        assert spec["trialTemplate"]["primaryContainerName"] == trainer_constants.NODE
        assert len(spec["trialTemplate"]["trialParameters"]) == len(
            test_case.config["search_space"]
        )
        assert spec["trialTemplate"]["trialSpec"]["apiVersion"] == trainer_constants.API_VERSION
        assert spec["trialTemplate"]["trialSpec"]["kind"] == trainer_constants.TRAINJOB_KIND

        expected_trial_config = TrialConfig()
        assert spec["maxTrialCount"] == expected_trial_config.num_trials
        assert spec["parallelTrialCount"] == expected_trial_config.parallel_trials

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="single search space parameter",
            expected_status=SUCCESS,
            config={
                "search_space": {
                    "lr": Search.uniform(min=0.001, max=0.1),
                },
            },
        ),
        TestCase(
            name="multiple search space parameters",
            expected_status=SUCCESS,
            config={
                "search_space": {
                    "lr": Search.uniform(min=0.001, max=0.1),
                    "epochs": Search.choice([10, 20, 30]),
                },
            },
        ),
    ],
)
def test_optimize_no_input_mutation(optimizer_backend, test_case):
    """Test that optimize() does not mutate the caller's input objects."""
    print("Executing test:", test_case.name)

    search_space = test_case.config["search_space"]

    original_names = {
        param_name: param_spec.name for param_name, param_spec in search_space.items()
    }

    trial_template = TrainJobTemplate(
        trainer=CustomTrainer(
            func=lambda: None,
            func_args={"existing_arg": "original_value"},
            num_nodes=1,
        ),
    )
    original_func_args = dict(trial_template.trainer.func_args)

    try:
        optimizer_backend.optimize(
            trial_template=trial_template,
            search_space=search_space,
        )

        assert test_case.expected_status == SUCCESS

        for param_name, param_spec in search_space.items():
            assert param_spec.name == original_names[param_name]

        assert trial_template.trainer.func_args == original_func_args

    except Exception as e:
        assert test_case.expected_status != SUCCESS
        assert isinstance(e, test_case.expected_error)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with multiple jobs",
            expected_status=SUCCESS,
            config={},
        ),
        TestCase(
            name="timeout error when listing jobs",
            expected_status=FAILED,
            config={"namespace": TIMEOUT},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when listing jobs",
            expected_status=FAILED,
            config={"namespace": RUNTIME},
            expected_error=RuntimeError,
        ),
    ],
)
def test_list_jobs(kubernetes_backend, test_case):
    """Test KubernetesBackend.list_jobs returns expected OptimizationJob list."""
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        jobs = kubernetes_backend.list_jobs()

        assert test_case.expected_status == SUCCESS
        assert isinstance(jobs, list)
        assert len(jobs) == 2
        assert all(isinstance(j, OptimizationJob) for j in jobs)
        assert jobs[0].name == "opt-job-1"
        assert jobs[1].name == "opt-job-2"
        for job in jobs:
            assert job.status == constants.OPTIMIZATION_JOB_COMPLETE
            assert asdict(job.algorithm) == asdict(RandomSearch())
            assert len(job.trials) == 1

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with existing job",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME},
        ),
        TestCase(
            name="timeout error when getting job",
            expected_status=FAILED,
            config={"name": TIMEOUT},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when getting job",
            expected_status=FAILED,
            config={"name": RUNTIME},
            expected_error=RuntimeError,
        ),
    ],
)
def test_get_job(kubernetes_backend, test_case):
    """Test KubernetesBackend.get_job with success and error paths."""
    print("Executing test:", test_case.name)
    try:
        job = kubernetes_backend.get_job(**test_case.config)

        assert test_case.expected_status == SUCCESS
        assert isinstance(job, OptimizationJob)
        assert job.name == BASIC_OPTIMIZATION_JOB_NAME
        assert job.status == constants.OPTIMIZATION_JOB_COMPLETE
        assert job.creation_timestamp == CREATION_TIMESTAMP
        assert asdict(job.algorithm) == asdict(RandomSearch())
        assert asdict(job.objectives[0]) == asdict(
            Objective(metric="loss", direction=Direction.MINIMIZE)
        )
        assert job.search_space == {
            "learning_rate": ContinuousSearchSpace(
                min=0.001, max=0.1, distribution=Distribution.UNIFORM
            ),
        }
        assert job.trial_config.num_trials == 10
        assert job.trial_config.parallel_trials == 1
        assert len(job.trials) == 1
        assert job.trials[0].name == BASIC_TRIAL_NAME
        assert job.trials[0].parameters == {"learning_rate": "0.01"}
        assert asdict(job.trials[0].metrics[0]) == asdict(
            Metric(name="loss", latest="0.05", max="0.1", min="0.02")
        )

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with best trial logs",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME},
            expected_output=["test log content"],
        ),
        TestCase(
            name="valid flow with explicit trial name",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "trial_name": BASIC_TRIAL_NAME},
            expected_output=["test log content"],
        ),
        TestCase(
            name="no trials returns empty",
            expected_status=SUCCESS,
            config={"name": "empty-trials"},
            expected_output=[],
        ),
    ],
)
def test_get_job_logs(kubernetes_backend, test_case):
    """Test KubernetesBackend.get_job_logs for best and explicit trials."""
    print("Executing test:", test_case.name)
    try:
        if test_case.config.get("name") == "empty-trials":
            # Mock _get_best_trial to return None and get_job to return job with no trials.
            kubernetes_backend._get_best_trial = Mock(return_value=None)
            mock_job = Mock()
            mock_job.trials = []
            kubernetes_backend.get_job = Mock(return_value=mock_job)

        logs = kubernetes_backend.get_job_logs(
            name=test_case.config["name"],
            trial_name=test_case.config.get("trial_name"),
        )
        logs_list = list(logs)

        assert test_case.expected_status == SUCCESS
        assert logs_list == test_case.expected_output

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with best trial",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME},
            expected_output=Result(
                parameters={"learning_rate": "0.01"},
                metrics=[Metric(name="loss", latest="0.05", max="0.1", min="0.02")],
            ),
        ),
        TestCase(
            name="no best trial returns None",
            expected_status=SUCCESS,
            config={"name": "no-best-trial"},
            expected_output=None,
        ),
    ],
)
def test_get_best_results(kubernetes_backend, test_case):
    """Test KubernetesBackend.get_best_results returns optimal trial Result."""
    print("Executing test:", test_case.name)
    try:
        if test_case.config.get("name") == "no-best-trial":
            # Return experiment with no currentOptimalTrial.
            no_best_exp = get_experiment(name="no-best-trial")
            no_best_exp.status.current_optimal_trial = None
            mock_thread = Mock()
            mock_thread.get.return_value = no_best_exp.to_dict()
            kubernetes_backend.custom_api.get_namespaced_custom_object = Mock(
                return_value=mock_thread
            )

        result = kubernetes_backend.get_best_results(**test_case.config)

        assert test_case.expected_status == SUCCESS
        if test_case.expected_output is None:
            assert result is None
        else:
            assert isinstance(result, Result)
            assert asdict(result) == asdict(test_case.expected_output)

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="wait for complete status",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME},
        ),
        TestCase(
            name="wait for multiple statuses",
            expected_status=SUCCESS,
            config={
                "name": BASIC_OPTIMIZATION_JOB_NAME,
                "status": {
                    constants.OPTIMIZATION_JOB_RUNNING,
                    constants.OPTIMIZATION_JOB_COMPLETE,
                },
            },
        ),
        TestCase(
            name="invalid status set error",
            expected_status=FAILED,
            config={
                "name": BASIC_OPTIMIZATION_JOB_NAME,
                "status": {"InvalidStatus"},
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="polling interval more than timeout error",
            expected_status=FAILED,
            config={
                "name": BASIC_OPTIMIZATION_JOB_NAME,
                "timeout": 1,
                "polling_interval": 2,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="job failed unexpectedly",
            expected_status=FAILED,
            config={
                "name": "failed-job",
                "status": {constants.OPTIMIZATION_JOB_RUNNING},
            },
            expected_error=RuntimeError,
        ),
        TestCase(
            name="timeout waiting for status",
            expected_status=FAILED,
            config={
                "name": BASIC_OPTIMIZATION_JOB_NAME,
                "status": {constants.OPTIMIZATION_JOB_FAILED},
                "polling_interval": 1,
                "timeout": 2,
            },
            expected_error=TimeoutError,
        ),
    ],
)
def test_wait_for_job_status(kubernetes_backend, test_case):
    """Test KubernetesBackend.wait_for_job_status polling and validation."""
    print("Executing test:", test_case.name)

    original_get_job = kubernetes_backend.get_job

    def mock_get_job(name):
        job = original_get_job(name)
        if test_case.config.get("name") == "failed-job":
            job.status = constants.OPTIMIZATION_JOB_FAILED
        return job

    kubernetes_backend.get_job = mock_get_job

    try:
        job = kubernetes_backend.wait_for_job_status(**test_case.config)

        assert test_case.expected_status == SUCCESS
        assert isinstance(job, OptimizationJob)
        assert job.status in test_case.config.get("status", {constants.OPTIMIZATION_JOB_COMPLETE})

    except Exception as e:
        assert type(e) is test_case.expected_error

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with deletion",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME},
        ),
        TestCase(
            name="timeout error when deleting job",
            expected_status=FAILED,
            config={"namespace": TIMEOUT},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="runtime error when deleting job",
            expected_status=FAILED,
            config={"namespace": RUNTIME},
            expected_error=RuntimeError,
        ),
    ],
)
def test_delete_job(kubernetes_backend, test_case):
    """Test KubernetesBackend.delete_job removes Experiment CR."""
    print("Executing test:", test_case.name)
    try:
        kubernetes_backend.namespace = test_case.config.get("namespace", DEFAULT_NAMESPACE)
        kubernetes_backend.delete_job(test_case.config.get("name"))
        assert test_case.expected_status == SUCCESS

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid flow with Experiment and Trial events",
            expected_status=SUCCESS,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME},
            expected_output=[
                Event(
                    involved_object_kind=constants.EXPERIMENT_KIND,
                    involved_object_name=BASIC_OPTIMIZATION_JOB_NAME,
                    message="Experiment created successfully",
                    reason="Created",
                    event_time=datetime.datetime(2025, 6, 1, 10, 30, 0),
                ),
                Event(
                    involved_object_kind=constants.TRIAL_KIND,
                    involved_object_name=BASIC_TRIAL_NAME,
                    message="Trial started",
                    reason="Running",
                    event_time=datetime.datetime(2025, 6, 1, 10, 31, 0),
                ),
            ],
        ),
        TestCase(
            name="timeout error from get_job propagates",
            expected_status=FAILED,
            config={"name": TIMEOUT},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="timeout error when listing events",
            expected_status=FAILED,
            config={"name": BASIC_OPTIMIZATION_JOB_NAME, "events_timeout": True},
            expected_error=TimeoutError,
        ),
    ],
)
def test_get_job_events(kubernetes_backend, test_case):
    """Test KubernetesBackend.get_job_events filters Experiment and Trial events."""
    print("Executing test:", test_case.name)
    try:
        # Override list_namespaced_event to simulate timeout on .get().
        if test_case.config.get("events_timeout"):
            timeout_thread = Mock()
            timeout_thread.get.side_effect = multiprocessing.TimeoutError()
            kubernetes_backend.core_api.list_namespaced_event = Mock(return_value=timeout_thread)

        events = kubernetes_backend.get_job_events(name=test_case.config["name"])

        assert test_case.expected_status == SUCCESS
        assert isinstance(events, list)
        assert len(events) == len(test_case.expected_output)
        assert [asdict(e) for e in events] == [asdict(e) for e in test_case.expected_output]

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")
