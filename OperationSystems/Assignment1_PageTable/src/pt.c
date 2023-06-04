#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <err.h>
#include <sys/mman.h>
#include "os.h"

#define NPAGES	(1024*1024)
#define PAGE_ENTRY_SIZE (512)
#define NLEVELS (5)
#define START_BIT_OF_VPN_IN_VA (12)
#define START_BIT_OF_PPN_IN_PA (12)
#define SIZE_OF_VPN (45)

static char* pages[NPAGES];

uint64_t alloc_page_frame(void) {
	static uint64_t nalloc;
	uint64_t ppn;
	void* va;

	if (nalloc == NPAGES)
		errx(1, "out of physical memory");

	/* OS memory management isn't really this simple */
	ppn = nalloc;
	nalloc++;

	va = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
	if (va == MAP_FAILED)
		err(1, "mmap failed");

	pages[ppn] = va;	
	return ppn;
}

void* phys_to_virt(uint64_t phys_addr) {
	uint64_t ppn = phys_addr >> 12;
	uint64_t off = phys_addr & 0xfff;
	char* va = NULL;

	if (ppn < NPAGES)
		va = pages[ppn] + off;

	return va;
}

void page_table_update(uint64_t pt, uint64_t vpn, uint64_t ppn){
    int level = 1;
    int size_of_levelVpn = SIZE_OF_VPN / NLEVELS;
    uint64_t new_ppn, new_entry, part_vpn, p_entry;
    uint64_t *pt_pointer;

    p_entry = (pt << START_BIT_OF_PPN_IN_PA); /* make pt as a full physical address*/
    pt_pointer = phys_to_virt(p_entry); /* A pointer to the root of PT */
        
    for (; level < NLEVELS; level++){
        part_vpn = extract_bits(vpn, size_of_levelVpn*(NLEVELS - level) , size_of_levelVpn);
        p_entry = *(pt_pointer + part_vpn);
                
        if (p_entry % 2 == 0){ /* The mapping is invalid (valid bit = 0) */
            if (ppn != NO_MAPPING){ /* We neet to create new frame for this map */
                new_ppn = alloc_page_frame();
                p_entry = (new_ppn << START_BIT_OF_PPN_IN_PA) + 0x1; 
                                    /* adding 00--00 in the last 12 bits of PA */
                *(pt_pointer + part_vpn) = p_entry;
            }
            else{ /* We need to destroy mapping that is already not valid */
                return;
            }
        }
        
        pt_pointer = phys_to_virt((p_entry >> 12) << 12); /* Update the pointer to the next 
                                                        pgae in the next level. Set 00--00
                                                        in the 12 bits in the end */
    }
    
    /* LAST LEVEL OF MAPPING */
    part_vpn = extract_bits(vpn, 0 , size_of_levelVpn);
    
    if (ppn != NO_MAPPING){ /* Creating the last mapping */
        new_entry = (ppn << START_BIT_OF_PPN_IN_PA) + 0x1; /* adding 00--01 in
                                                         the last 12 bits of PA */
        *(pt_pointer + part_vpn) = new_entry;
    }
    else { /* Destroy the mapping to the given vpn*/
        *(pt_pointer + part_vpn) = ((*(pt_pointer + part_vpn)) >> 1) << 1; /* Set 0 invalid bit */
    }
}

uint64_t page_table_query(uint64_t pt, uint64_t vpn){
    int level = 1;
    uint64_t ppn, part_vpn, p_entry;
    int size_of_levelVpn = SIZE_OF_VPN / NLEVELS;
    uint64_t *pt_pointer;

    p_entry = (pt << START_BIT_OF_PPN_IN_PA); /* make pt as a full physical address*/
    pt_pointer = phys_to_virt(p_entry); /* A pointer to the root of PT */

    for (; level <= NLEVELS; level++){
        part_vpn = extract_bits(vpn, size_of_levelVpn*(NLEVELS - level) , size_of_levelVpn);
        p_entry = *(pt_pointer + part_vpn);
                
        if (p_entry % 2 == 0){ /* The mapping is invalid (valid bit = 0) */
            return NO_MAPPING;
        }
 
        if (level < NLEVELS){ 
            pt_pointer = phys_to_virt((p_entry >> 12) << 12); /* update the pointer
                         to the next pgae in the next level. Set 00-00 in the end of the addres */
        }
    }
    ppn = extract_bits(p_entry, START_BIT_OF_PPN_IN_PA, -1);
    
    return ppn;
}

uint64_t extract_bits(uint64_t w, int start, int size){
    /* Extract the (size) bits from (start) position (0 from right) 
        in (w) and returns the result as unit64_t.
        Assumptions: start+size < |w_(2)|=64, size > 0 
        if size = -1, the cut will be fro, start to the end of w */

        int i = 0;
        uint64_t ext = w >> start;
        uint64_t mask = 0;
        
        if (size == -1){
            size = 64 - start;
        }

        /* mask = 11...11 size times */
        for (; i < size; i++){
            mask = mask*2 + 1;
        }

        ext = ext & mask;
        return ext;
}
