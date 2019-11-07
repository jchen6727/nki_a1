#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _ch_CavL_reg(void);
extern void _ch_CavN_reg(void);
extern void _ch_KCaS_reg(void);
extern void _ch_Kdrfastngf_reg(void);
extern void _ch_KvAngf_reg(void);
extern void _ch_KvCaB_reg(void);
extern void _ch_Navngf_reg(void);
extern void _ch_leak_reg(void);
extern void _h_migliore_reg(void);
extern void _iconc_Ca_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," ch_CavL.mod");
    fprintf(stderr," ch_CavN.mod");
    fprintf(stderr," ch_KCaS.mod");
    fprintf(stderr," ch_Kdrfastngf.mod");
    fprintf(stderr," ch_KvAngf.mod");
    fprintf(stderr," ch_KvCaB.mod");
    fprintf(stderr," ch_Navngf.mod");
    fprintf(stderr," ch_leak.mod");
    fprintf(stderr," h_migliore.mod");
    fprintf(stderr," iconc_Ca.mod");
    fprintf(stderr, "\n");
  }
  _ch_CavL_reg();
  _ch_CavN_reg();
  _ch_KCaS_reg();
  _ch_Kdrfastngf_reg();
  _ch_KvAngf_reg();
  _ch_KvCaB_reg();
  _ch_Navngf_reg();
  _ch_leak_reg();
  _h_migliore_reg();
  _iconc_Ca_reg();
}
