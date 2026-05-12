#include "am_mcu_apollo.h"

#include "am_bsp.h"
#include "am_util.h"

#include "mnist.h"

#ifdef MEM
#include "mem.h"
#elif defined(VIRT)
#include "vfff.h"
#else
#include "fff.h"
#endif

#ifdef MEM
volatile uint32_t g_ReadCount = 0;
volatile uint32_t g_WriteCount = 0;
volatile uint32_t g_ReadWriteCount = 0;
#endif

#ifdef TIMING
#define TIMER_NUM 5
volatile uint32_t g_TimerCount = 0;
uint32_t timer_init(uint32_t ui32TimerNum) {
    am_hal_timer_config_t       TimerConfig;
    uint32_t ui32Status         = AM_HAL_STATUS_SUCCESS;

    // Set the timer configuration, the default timer configuration is HFRC_DIV16, EDGE, compares=0, no trig.
    am_hal_timer_default_config_set(&TimerConfig);
    ui32Status = am_hal_timer_config(ui32TimerNum, &TimerConfig);
    if ( ui32Status != AM_HAL_STATUS_SUCCESS )
    {
        am_util_stdio_printf("Failed to configure TIMER%d, return value=%d\n", ui32TimerNum, ui32Status);
        return ui32Status;
    }

    // Stop and clear the timer.
    am_hal_timer_clear(ui32TimerNum);
    am_hal_timer_stop(ui32TimerNum);

    // Timer interrupt not needed for this purpose.
    return ui32Status;
} // timer_init()
#endif
 

// Main
int main(void) {
#ifdef TIMING
    timer_init(TIMER_NUM);
    // Set the default cache configuration
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();
#endif
#ifndef HIGH_PERF
    am_bsp_low_power_init();
    am_hal_pwrctrl_mcu_mode_select(AM_HAL_PWRCTRL_MCU_MODE_LOW_POWER);
#else
    am_hal_pwrctrl_mcu_mode_select(AM_HAL_PWRCTRL_MCU_MODE_HIGH_PERFORMANCE);
#endif
#ifdef LED
    // Initialize LED0 for status indication
    am_hal_gpio_pinconfig(AM_BSP_GPIO_LED0, am_hal_gpio_pincfg_output);
    am_hal_gpio_state_write(AM_BSP_GPIO_LED0, AM_HAL_GPIO_OUTPUT_CLEAR);
#endif
    while (1)
    {
#ifdef TIMING
        am_hal_timer_clear(TIMER_NUM);   // The clear function also starts the timer
#endif
        for (int i = 0; i < 100; i++) {
            MODEL();
        }

#ifdef TIMING
        am_hal_timer_stop(TIMER_NUM);
        g_TimerCount = am_hal_timer_read(TIMER_NUM);
#endif
#ifdef LED
        am_hal_gpio_state_write(AM_BSP_GPIO_LED0, AM_HAL_GPIO_OUTPUT_TOGGLE);
#endif
        // Go to Deep Sleep.
        am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
    }
}
