// SPDX-License-Identifier: GPL-2.0-only
/* Temporary fixed-voltage test driver. The image omits PM_OPP, so cpufreq-dt
 * cannot probe. Use the standard CPUfreq/CCF paths with their timer notifiers;
 * accept only the board's two existing equal-voltage OPPs. No MMIO writes. */
#include <linux/module.h>
#include <linux/cpu.h>
#include <linux/cpufreq.h>
#include <linux/clk.h>
#include <linux/of.h>
static struct clk *cpu_clk;
static u32 opps[4];
static struct cpufreq_frequency_table table[]={
 {.frequency=325000},{.frequency=650000},{.frequency=CPUFREQ_TABLE_END}
};
static int target(struct cpufreq_policy *p,unsigned int index)
{
 unsigned long hz;
 if(index>1)return -EINVAL;
 hz=table[index].frequency*1000ul;
 if(clk_round_rate(cpu_clk,hz)!=hz)return -EINVAL;
 return clk_set_rate(cpu_clk,hz);
}
static int policy_init(struct cpufreq_policy *p)
{
 p->clk=cpu_clk;
 cpufreq_generic_init(p,table,100000);
 return 0;
}
static struct cpufreq_driver driver={
 .name="mlp-fixed",.flags=CPUFREQ_NEED_INITIAL_FREQ_CHECK,
 .verify=cpufreq_generic_frequency_table_verify,
 .target_index=target,.get=cpufreq_generic_get,.init=policy_init,
 .attr=cpufreq_generic_attr
};
static int __init start(void)
{
 struct device *dev=get_cpu_device(0);
 int rc;
 if(!dev||!of_machine_is_compatible("xlnx,zynq-7000"))return -ENODEV;
 if(of_property_count_u32_elems(dev->of_node,"operating-points")!=4)return -EINVAL;
 rc=of_property_read_u32_array(dev->of_node,"operating-points",opps,4);
 if(rc)return rc;
 if(opps[0]!=650000||opps[2]!=325000||opps[1]!=opps[3]||opps[1]!=1000000)return -EINVAL;
 cpu_clk=clk_get(dev,NULL);if(IS_ERR(cpu_clk))return PTR_ERR(cpu_clk);
 if(clk_get_rate(cpu_clk)!=650000000ul){clk_put(cpu_clk);return -EINVAL;}
 rc=cpufreq_register_driver(&driver);
 if(rc)clk_put(cpu_clk);
 return rc;
}
static void __exit stop(void)
{
 cpufreq_unregister_driver(&driver);
 if(clk_get_rate(cpu_clk)!=650000000ul)
  pr_warn("MLP CPU test module unloaded at a non-default CPU clock\n");
 clk_put(cpu_clk);
}
module_init(start);module_exit(stop);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Temporary Zynq650/325MHz equal-voltage CPUfreq test driver");
