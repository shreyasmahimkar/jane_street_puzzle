#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CommonCrypto/CommonDigest.h>
#include <dispatch/dispatch.h>

#define MAX_WORDS 500000

char *words[MAX_WORDS];
int word_lens[MAX_WORDS];
int num_words = 0;

// target hash bytes: c7ef65233c40aa32c2b9ace37595fa7c
unsigned char target_hash[16] = {
    0xc7, 0xef, 0x65, 0x23, 0x3c, 0x40, 0xaa, 0x32, 
    0xc2, 0xb9, 0xac, 0xe3, 0x75, 0x95, 0xfa, 0x7c
};

int main(int argc, char **argv) {
    FILE *f = fopen("words_alpha.txt", "r");
    if (!f) {
        printf("Could not open words_alpha.txt\n");
        return 1;
    }
    char buf[256];
    while (fgets(buf, sizeof(buf), f)) {
        size_t len = strlen(buf);
        while (len > 0 && (buf[len-1] == '\r' || buf[len-1] == '\n')) {
            buf[len-1] = '\0';
            len--;
        }
        if (len > 0) {
            words[num_words] = strdup(buf);
            word_lens[num_words] = len;
            num_words++;
            if (num_words >= MAX_WORDS) break;
        }
    }
    fclose(f);
    printf("Loaded %d words.\n", num_words);

    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    
    __block int found = 0;
    
    dispatch_apply(num_words, queue, ^(size_t i) {
        if (found) return; // fast exit
        
        char buffer[1024];
        unsigned char md5[16];
        
        int len1 = word_lens[i];
        const char *w1 = words[i];
        memcpy(buffer, w1, len1);
        buffer[len1] = ' ';
        
        for (int j = 0; j < num_words; j++) {
            if (found) return;
            int len2 = word_lens[j];
            
            memcpy(buffer + len1 + 1, words[j], len2);
            int total_len = len1 + 1 + len2;
            
            CC_MD5(buffer, total_len, md5);
            
            int match = 1;
            for(int k=0; k<16; k++) {
                if (md5[k] != target_hash[k]) {
                    match = 0;
                    break;
                }
            }
            if (match) {
                buffer[total_len] = '\0';
                printf("FOUND MATCH: %s\n", buffer);
                found = 1;
                exit(0);
            }
        }
    });
    
    if (!found) {
        printf("Not found!\n");
    }
    
    return 0;
}
