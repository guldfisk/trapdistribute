import os
import random
import statistics
import time
import typing as t
from enum import Enum

from promise import Promise

from proxypdf.write import save_proxy_pdf

from mtgorp.db.load import Loader
from mtgorp.utilities.containers import HashableMultiset

from mtgimg.load import Loader as ImageLoader

from magiccube.laps.lap import Lap
from magiccube.laps.traps.trap import IntentionType
from magiccube.collections.delta import CubeDelta

from misccube import paths
from misccube.cubeload.load import CubeLoader
from misccube.trapification import algorithm
from misccube.trapification.algorithm import Distributor, DeltaDistributor, ConstraintSetBluePrint
from misccube.trapification.fetch import ConstrainedNodeFetcher
from misccube.trapification.persist import TrapCollectionPersistor, TrapCollection


GARBAGE_OUT_PATH = os.path.join(paths.OUT_DIR, 'garbage_distribution.pdf')
GARBAGE_NEW_OUT_PATH = os.path.join(paths.OUT_DIR, 'garbage_distribution_new.pdf')
GARBAGE_REMOVED_OUT_PATH = os.path.join(paths.OUT_DIR, 'garbage_distribution_removed.pdf')

GARBAGE_LANDS_OUT_PATH = os.path.join(paths.OUT_DIR, 'garbage_lands_distribution.pdf')
GARBAGE_LANDS_NEW_OUT_PATH = os.path.join(paths.OUT_DIR, 'garbage_lands_distribution_new.pdf')
GARBAGE_LANDS_REMOVED_OUT_PATH = os.path.join(paths.OUT_DIR, 'garbage_lands_distribution_removed.pdf')

BOTH_OUT_PATH = os.path.join(paths.OUT_DIR, 'full_garbage_distribution.pdf')
BOTH_NEW_OUT_PATH = os.path.join(paths.OUT_DIR, 'full_garbage_distribution_new.pdf')
BOTH_REMOVED_OUT_PATH = os.path.join(paths.OUT_DIR, 'full_garbage_distribution_removed.pdf')


class IntentionTypeTarget(Enum):
	GARBAGE = 'garbage'
	LANDS_GARBAGE = 'lands_garbage'
	BOTH = 'both'


def proxy_laps(
	laps: t.Iterable[Lap],
	image_loader: ImageLoader,
	file_name: str,
	margin_size: float = .1,
	card_margin_size: float = .01,
) -> None:
	promises = tuple(
		image_loader.get_image(lap, save=False)
		for lap in
		laps
	)

	images = Promise.all(
		promises
	).get()

	save_proxy_pdf(
		file = file_name,
		images = images,
		margin_size = margin_size,
		card_margin_size = card_margin_size,
	)


_GROUP_WEIGHTS = {
	'WHITE': 1,
	'BLUE': 1.5,
	'BLACK': 1,
	'RED': 1,
	'GREEN': 1,
	'drawgo': 3,
	'mud': 3,
	'post': 4,
	'midrange': 2,
	'mill': 4,
	'reanimate': 4,
	'burn': 4,
	'hatebear': 2,
	'removal': 1,
	'lock': 3,
	'yardvalue': 3,
	'ld': 3,
	'storm': 4,
	'tezz': 3,
	'lands': 3,
	'shatter': 3,
	'bounce': 3,
	'shadow': 4,
	'stifle': 4,
	'beat': 1,
	'cheat': 4,
	'pox': 3,
	'counter': 3,
	'discard': 2,
	'cantrip': 4,
	'balance': 3,
	'stasis': 4,
	'standstill': 3,
	'whitehate': 4,
	'bluehate': 4,
	'blackhate': 4,
	'redhate': 4,
	'greenhate': 4,
	'antiwaste': 4,
	'delirium': 3,
	'sacvalue': 2,
	'lowtoughnesshate': 4,
	'armageddon': 4,
	'stax': 3,
	'bloom': 3,
	'weldingjar': 3,
	'drawhate': 4,
	'pluscard': 3,
	'ramp': 3,
	'devoteddruid': 4,
	'fetchhate': 4,
	'dragon': 2,
	'company': 2,
	'naturalorder': 3,
	'flash': 3,
	'wincon': 3,
	'vial': 4,
	# lands
	'fixing': 3,
	'colorlessvalue': 1,
	'fetchable': 2,
	'indestructable': 4,
	'legendarymatters': 1,
	'sol': 3,
	'manland': 4,
	'storage': 3,
	'croprotate': 3,
	'dnt': 3,
	'equipment': 4,
	'livingdeath': 3,
	'eggskci': 3,
	'hightide': 3,
	'fatty': 3,
	'walker': 4,
	'blink': 2,
	'miracles': 3,
	'city': 4,
	'wrath': 2,
	'landtax': 4,
	'discardvalue': 2,
	'edict': 2,
	'phoenix': 4,
	'enchantress': 2,
	'dork': 2,
	'tinker': 3,
	'highpowerhate': 2,
	'affinity': 3,
	'academy': 4,
	'stompy': 2,
	'shardless': 3,
	'lanterns': 3,
	'depths': 3,
	'survival': 2,
	'landstill': 2,
	'moat': 4,
	'combo': 3,
	'kite': 3,
	'haste': 3,
	'fog': 3,
}


GROUP_WEIGHT_EXPONENT = 1.5


GROUP_WEIGHTS = {key: value ** GROUP_WEIGHT_EXPONENT for key, value in _GROUP_WEIGHTS.items()}


def calculate(
	generations: int,
	trap_amount: int,
	max_delta: t.Optional[int] = None,
	create_proxy_pdfs: bool = True,
):
	random.seed()

	db = Loader.load()
	image_loader = ImageLoader()
	fetcher = ConstrainedNodeFetcher(db)
	cube_loader = CubeLoader(db)

	trap_collection_persistor = TrapCollectionPersistor(db)

	constrained_nodes = fetcher.fetch_garbage()

	print(f'loaded {len(constrained_nodes)} nodes')

	cube = cube_loader.load()

	cube_traps = HashableMultiset(
		trap
		for trap in
		cube.traps
		if (
			trap.intention_type == IntentionType.GARBAGE
			or trap.intention_type == IntentionType.LAND_GARBAGE
		)
	)

	blue_print = ConstraintSetBluePrint(
		(
			algorithm.ValueDistributionHomogeneityConstraint,
			2,
			{},
		),
		(
			algorithm.GroupExclusivityConstraint,
			2,
			{'group_weights': GROUP_WEIGHTS},
		),
		(
			algorithm.SizeHomogeneityConstraint,
			1,
			{},
		),
	)

	if max_delta is not None and max_delta > 0:
		distributor = DeltaDistributor(
			constrained_nodes = constrained_nodes,
			trap_amount = trap_amount,
			origin_trap_collection = cube_traps,
			constraint_set_blue_print = blue_print,
			max_trap_delta = max_delta,
			mate_chance = .45,
			mutate_chance = .35,
			tournament_size = 3,
			population_size = 600,
		)
	else:
		distributor = Distributor(
			constrained_nodes = constrained_nodes,
			trap_amount = trap_amount,
			constraint_set_blue_print = blue_print,
			mate_chance = .5,
			mutate_chance = .45,
			tournament_size = 4,
			population_size = 600,
		)

	random_fitness = statistics.mean(
		map(distributor.constraint_set.total_score, distributor.sample_random_population)
	)

	st = time.time()

	winner = distributor.evaluate(generations).best

	print(f'Done in {time.time() - st} seconds')

	print('Random fitness:', random_fitness)

	try:
		print('Current cube fitness:', distributor.evaluate_cube(cube_traps))
	except ValueError:
		print('Nodes does not match current cube')
		_, added, removed = distributor.trap_collection_to_trap_distribution(cube_traps, constrained_nodes)
		print('added:', added)
		print('removed:', removed)

	print('Winner fitness:', winner.fitness.values[0])

	distributor.show_plot()

	winner_traps = winner.as_trap_collection
	for trap in winner_traps:
		trap._intention_type = IntentionType.GARBAGE

	new_traps = winner_traps - cube_traps
	removed_traps = cube_traps - winner_traps

	print('New traps', len(new_traps))

	trap_collection = TrapCollection(winner_traps)

	trap_collection_persistor.persist(trap_collection)

	print('\n------------------------------------------------\n')
	print(trap_collection.minimal_string_list)
	print('\n------------------------------------------------\n')

	print('traps persisted')

	if create_proxy_pdfs:
		out, new_out, removed_out = GARBAGE_OUT_PATH, GARBAGE_NEW_OUT_PATH, GARBAGE_REMOVED_OUT_PATH

		proxy_laps(
			laps = winner_traps,
			image_loader = image_loader,
			file_name = out,
		)

		proxy_laps(
			laps = new_traps,
			image_loader = image_loader,
			file_name = new_out,
		)

		proxy_laps(
			laps = removed_traps,
			image_loader = image_loader,
			file_name = removed_out,
		)

		print('proxying done')


def main():
	trap_amount = 108

	calculate(
		generations = 5,
		trap_amount = trap_amount,
		max_delta = 0,
		create_proxy_pdfs = False,
	)



if __name__ == '__main__':
	main()