"""Setup and running of the openai es optimization program."""

import logging
from random import Random

from sqlalchemy.ext.asyncio import AsyncSession

from optimizer import Optimizer
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import DbId
from revolve2.standard_resources.modular_robots import gecko

import optuna
from sqlalchemy.future import select
from revolve2.core.optimization.ea.openai_es import DbOpenaiESOptimizerIndividual
from revolve2.core.database.serializers import Ndarray1xnSerializer


async def objective(trial, t, rng):
    """Run the optimization process."""
    POPULATION_SIZE = trial.suggest_int('population_size', 5, 20)  # was: 10
    SIGMA = trial.suggest_float('sigma', 0.001, 0.01)  # was: 0.1
    LEARNING_RATE = trial.suggest_float('learning_rate', 0.001, 0.5)  # was: 0.05
    NUM_GENERATIONS = 3  # trial.suggest_int('num_generations', 2, 10)  # was: 3

    SIMULATION_TIME = 10
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 60

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    print(f"trial {t}: {trial.params}")

    # database
    database = open_async_database_sqlite("./database", t, create=True)

    # unique database identifier for optimizer
    db_id = DbId.root("optuna")

    body = gecko()

    maybe_optimizer = await Optimizer.from_database(
        database=database,
        db_id=db_id,
        rng=rng,
        robot_body=body,
        simulation_time=SIMULATION_TIME,
        sampling_frequency=SAMPLING_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
        num_generations=NUM_GENERATIONS,
    )
    if maybe_optimizer is not None:
        logging.info(
            f"Recovered. Last finished generation: {maybe_optimizer.generation_number}."
        )
        optimizer = maybe_optimizer
    else:
        logging.info("No recovery data found. Starting at generation 0.")
        optimizer = await Optimizer.new(
            database=database,
            db_id=db_id,
            rng=rng,
            population_size=POPULATION_SIZE,
            sigma=SIGMA,
            learning_rate=LEARNING_RATE,
            robot_body=body,
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
            num_generations=NUM_GENERATIONS,
        )

    logging.info("Starting optimization process..")

    await optimizer.run()

    # async with AsyncSession(database) as session:
    #     best_individual = (await session.execute(
    #         select(DbEAOptimizerIndividual, DbFloat)
    #         .filter(DbEAOptimizerIndividual.fitness_id == DbFloat.id)
    #         .order_by(DbFloat.value.desc())
    #     )).first()
    #     assert best_individual is not None
    #     best_fitness = best_individual[1].value

    async with AsyncSession(database) as session:
        best_individual = (
            (
                await session.execute(
                    select(DbOpenaiESOptimizerIndividual).order_by(
                        DbOpenaiESOptimizerIndividual.fitness.desc()
                    )
                )
            )
            .scalars()
            .all()[0]
        )

        params = [
            p
            for p in (
                await Ndarray1xnSerializer.from_database(
                    session, [best_individual.individual]
                )
            )[0]
        ]

        # print(f"fitness: {best_individual.fitness}")
        # print(f"params: {params}")

    logging.info("Finished optimizing.")
    print(f"best fitness: {best_individual.fitness}")
    return best_individual.fitness


async def main() -> None:
    study_name = "ES_500trials_v5"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
                                sampler=optuna.samplers.RandomSampler(), direction="maximize")
    n_trials = 500
    for t in range(n_trials):
        rng = Random()
        rng.seed(t)
        trial = study.ask()
        fitness = await objective(trial, t, rng)
        study.tell(trial, fitness)

    print("Best Trial: ", study.best_trial.number, " with fitness ", study.best_trial.value, " and params ",
          study.best_trial.params)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
