import wandb

name = "spider_turn1_truncate2048_0904"
run = wandb.init(project="AgentLightningValidate", name=name, id=name+"1")
run.log({"val/reward": 0.708}, step=0)
run.log({"val/reward_test": 0.6809501630181649, "val/reward_dev": 0.6112185686653772}, step=0)
run.log({"val/reward": 0.796}, step=128)
run.log({"val/reward_test": 0.7992547741034001, "val/reward_dev": 0.7398452611218569}, step=128)
run.log({"val/reward": 0.824}, step=256)
run.log({"val/reward_test": 0.8020493712156498, "val/reward_dev": 0.7504835589941973}, step=256)
run.log({"val/reward": 0.824}, step=384)
run.log({"val/reward_test": 0.8122962272938985, "val/reward_dev": 0.7620889748549323}, step=384)
run.log({"val/reward": 0.806}, step=512)
run.log({"val/reward_test": 0.8146250582207731, "val/reward_dev": 0.7475822050290135}, step=512)
run.log({"val/reward": 0.824}, step=768)
run.log({"val/reward_test": 0.823474615742897, "val/reward_dev": 0.7601547388781431}, step=768)
run.log({"val/reward_test": 0.7512808570097811, "val/reward_dev": 0.7601547388781431}, step=896)
run.log({"val/reward_test": 0.7522123893805309, "val/reward_dev": 0.7495164410058027}, step=1024)
run.finish()


name = "spider_turn3_truncate2048_4321"
run = wandb.init(project="AgentLightningValidate", name=name, id=name+"1")

run.log({"val/reward": 0.688}, step=0)
run.log({"val/reward_test": 0.7023754075454122, "val/reward_dev": 0.6363636363636364}, step=0)
run.log({"val/reward": 0.772}, step=128)
run.log({"val/reward_test": 0.8039124359571496, "val/reward_dev": 0.7427466150870407}, step=128)
run.log({"val/reward": 0.772}, step=256)
run.log({"val/reward_test": 0.7936655798789007, "val/reward_dev": 0.7398452611218569}, step=256)
run.log({"val/reward": 0.776}, step=384)
run.log({"val/reward_test": 0.7945971122496507, "val/reward_dev": 0.7359767891682786}, step=384)
run.log({"val/reward": 0.802}, step=512)
run.log({"val/reward_test": 0.768048439683279, "val/reward_dev": 0.7040618955512572}, step=512)
run.log({"val/reward_test": 0.7908709827666511, "val/reward_dev": 0.741779497098646}, step=640)
run.log({"val/reward_test": 0.8164881229622729, "val/reward_dev": 0.7756286266924565}, step=768)
run.log({"val/reward_test": 0.8202142524452725, "val/reward_dev": 0.7823984526112185}, step=896)
# run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=1024)
run.finish()


name = "spider_turn5_truncate2048_0911"
run = wandb.init(project="AgentLightningValidate", name=name, id=name+"1")

run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=0)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=128)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=256)
run.log({"val/reward_test": 0.7373078714485328, "val/reward_dev": 0.7098646034816247}, step=384)
run.log({"val/reward_test": 0.7736376339077783, "val/reward_dev": 0.7282398452611218}, step=512)
run.log({"val/reward_test": 0.7685142058686539, "val/reward_dev": 0.6982591876208898}, step=768)
run.log({"val/reward_test": 0.7615277130880298, "val/reward_dev": 0.6808510638297872}, step=896)
run.log({"val/reward_test": 0.7615277130880298, "val/reward_dev": 0.6808510638297872}, step=1024)
run.finish()


# ===================================================
name = "spider_turn1_truncate4096_0904"
run = wandb.init(project="AgentLightningValidate", name=name, id=name+"1")
run.log({"val/reward": 0.708}, step=0)
run.log({"val/reward_test": 0.7135537959944108, "val/reward_dev": 0.6247582205029013}, step=0)
run.log({"val/reward": 0.796}, step=128)
run.log({"val/reward_test": 0.8015836050302748, "val/reward_dev": 0.7359767891682786}, step=128)
run.log({"val/reward": 0.824}, step=256)
run.log({"val/reward_test": 0.8220773171867722, "val/reward_dev": 0.7630560928433269}, step=256)
run.log({"val/reward": 0.824}, step=384)
run.log({"val/reward_test": 0.8248719142990218, "val/reward_dev": 0.7640232108317214}, step=384)
run.log({"val/reward": 0.806}, step=512)
run.log({"val/reward_test": 0.816022356776898, "val/reward_dev": 0.753384912959381}, step=512)
run.log({"val/reward": 0.824}, step=768)
run.log({"val/reward_test": 0.8220773171867722, "val/reward_dev": 0.7620889748549323}, step=768)

# from now on, we gurantee eval start after vllm start.
run.log({"val/reward_test": 0.8211457848160224, "val/reward_dev": 0.7543520309477756}, step=896) #~
run.log({"val/reward_test": 0.8230088495575221, "val/reward_dev": 0.7524177949709865}, step=1024) #~
run.finish()


name = "spider_turn3_truncate4096__0904"
run = wandb.init(project="AgentLightningValidate", name=name, id=name+"1")

run.log({"val/reward": 0.722}, step=0)
run.log({"val/reward_test": 0.735910572892408, "val/reward_dev": 0.6508704061895552}, step=0)
run.log({"val/reward": 0.802}, step=128)
run.log({"val/reward_test": 0.8164881229622729, "val/reward_dev": 0.7543520309477756}, step=128)
run.log({"val/reward": 0.814}, step=256)
run.log({"val/reward_test": 0.831858407079646, "val/reward_dev": 0.7707930367504836}, step=256)
run.log({"val/reward": 0.828}, step=384)
run.log({"val/reward_test": 0.8272007452258966, "val/reward_dev": 0.7843326885880078}, step=384)
run.log({"val/reward": 0.822}, step=512)
run.log({"val/reward_test": 0.8295295761527713, "val/reward_dev": 0.7640232108317214}, step=512)
run.log({"val/reward": 0.836}, step=768)
run.log({"val/reward_test": 0.8332557056357709, "val/reward_dev": 0.7756286266924565}, step=768)
# run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=896)
# run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=1024)
run.finish()


name = "spider_turn5_truncate4096_0911"
run = wandb.init(project="AgentLightningValidate", name=name, id=name+"1")

run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=0)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=128)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=256)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=384)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=512)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=768)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=896)
run.log({"val/reward_test": 0.7810898928737774, "val/reward_dev": 0.7243713733075435}, step=1024)
run.finish()


# ===================================================

name = "spider_check_turn2_truncate4096_0907"
run = wandb.init(project="AgentLightningValidate", name=name, id=name+"1")

run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=0)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=128)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=256)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=384)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=512)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=768)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=896)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=1024)
run.finish()

name = "spider_check_turn3_truncate4096_0907"
run = wandb.init(project="AgentLightningValidate", name=name, id=name+"1")

run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=0)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=128)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=256)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=384)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=512)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=768)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=896)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=1024)
run.finish()



name = "spider_check_turn5_truncate4096_0911"
run = wandb.init(project="AgentLightningValidate", name=name, id=name+"1")

run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=0)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=128)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=256)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=384)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=512)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=768)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=896)
run.log({"val/reward_test": 0., "val/reward_dev": 0.}, step=1024)
run.finish()