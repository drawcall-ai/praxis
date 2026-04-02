import { Command } from 'commander';
import { loadEnvUp, bold } from './cli/utils.js';
import { handleTrain } from './cli/train.js';
import { handleRun } from './cli/run.js';
import { handleValidate } from './cli/validate.js';
import { handleView } from './cli/view.js';

loadEnvUp(process.cwd());

const program = new Command()
  .name('praxis')
  .description('Define, train, and use optimized LLM prompts')
  .version('0.0.0');

program
  .command('train')
  .description('Optimize prompts from a definition file (auto-discovers via glob)')
  .option('-d, --definition <path>', 'definition file (default: auto-discover)')
  .option('-o, --output <path>', 'config output path (default: model.config.json next to definition)')
  .option('--split <ratio>', 'train/test split', '0.7')
  .option('-f, --force', 'skip version/schema guard and force retraining')
  .action(handleTrain);

program
  .command('run')
  .description('Run the model (auto-discovers definition and config)')
  .option('-d, --definition <path>', 'definition file (default: auto-discover)')
  .option('-c, --config <path>', 'config file (default: model.config.json next to definition)')
  .allowUnknownOption()
  .action((opts, cmd) => handleRun(opts, cmd.args));

program
  .command('view')
  .description('Launch a web UI to inspect eval runs and test manually')
  .option('-d, --definition <path>', 'definition file (default: auto-discover)')
  .option('-c, --config <path>', 'config file (default: model.config.json next to definition)')
  .option('-p, --port <port>', 'server port', '3473')
  .action(handleView);

program
  .command('validate')
  .description('Check that the config matches the definition schema')
  .option('-d, --definition <path>', 'definition file (default: auto-discover)')
  .option('-c, --config <path>', 'config file (default: model.config.json next to definition)')
  .action(handleValidate);

program.parseAsync().catch((err) => {
  console.error(`\n  ${bold('Error:')} ${err.message ?? err}\n`);
  process.exit(1);
});
