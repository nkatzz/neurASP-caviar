
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(br1gt_2,12,478). coords(br1gt_2,86,208,478).  visible(br1gt_2,478). happensAt(inactive(br1gt_2),478). orientation(br1gt_3,160,478). coords(br1gt_3,89,140,478).  visible(br1gt_3,478). happensAt(walking(br1gt_3),478).
orientation(br1gt_2,12,479). coords(br1gt_2,86,208,479).  visible(br1gt_2,479). happensAt(inactive(br1gt_2),479). orientation(br1gt_3,160,479). coords(br1gt_3,90,141,479).  visible(br1gt_3,479). happensAt(walking(br1gt_3),479).
orientation(br1gt_2,12,480). coords(br1gt_2,86,208,480).  visible(br1gt_2,480). happensAt(inactive(br1gt_2),480). orientation(br1gt_3,160,480). coords(br1gt_3,90,141,480).  visible(br1gt_3,480). happensAt(walking(br1gt_3),480).
orientation(br1gt_2,12,481). coords(br1gt_2,86,208,481).  visible(br1gt_2,481). happensAt(inactive(br1gt_2),481). orientation(br1gt_3,160,481). coords(br1gt_3,90,141,481).  visible(br1gt_3,481). happensAt(walking(br1gt_3),481).
orientation(br1gt_2,12,482). coords(br1gt_2,86,208,482).  visible(br1gt_2,482). happensAt(inactive(br1gt_2),482). orientation(br1gt_3,160,482). coords(br1gt_3,91,141,482).  visible(br1gt_3,482). happensAt(walking(br1gt_3),482).
orientation(br1gt_2,12,483). coords(br1gt_2,86,208,483).  visible(br1gt_2,483). happensAt(inactive(br1gt_2),483). orientation(br1gt_3,160,483). coords(br1gt_3,91,140,483).  visible(br1gt_3,483). happensAt(walking(br1gt_3),483).
orientation(br1gt_2,12,484). coords(br1gt_2,86,208,484).  visible(br1gt_2,484). happensAt(inactive(br1gt_2),484). orientation(br1gt_3,160,484). coords(br1gt_3,91,140,484).  visible(br1gt_3,484). happensAt(walking(br1gt_3),484).
orientation(br1gt_2,12,485). coords(br1gt_2,86,208,485).  visible(br1gt_2,485). happensAt(inactive(br1gt_2),485). orientation(br1gt_3,160,485). coords(br1gt_3,93,140,485).  visible(br1gt_3,485). happensAt(walking(br1gt_3),485).
orientation(br1gt_2,12,486). coords(br1gt_2,86,208,486).  visible(br1gt_2,486). happensAt(inactive(br1gt_2),486). orientation(br1gt_3,160,486). coords(br1gt_3,94,140,486).  visible(br1gt_3,486). happensAt(walking(br1gt_3),486).
orientation(br1gt_2,12,487). coords(br1gt_2,86,209,487).  visible(br1gt_2,487). happensAt(inactive(br1gt_2),487). orientation(br1gt_3,160,487). coords(br1gt_3,96,140,487).  visible(br1gt_3,487). happensAt(walking(br1gt_3),487).
orientation(br1gt_2,12,488). coords(br1gt_2,86,209,488).  visible(br1gt_2,488). happensAt(inactive(br1gt_2),488). orientation(br1gt_3,160,488). coords(br1gt_3,97,140,488).  visible(br1gt_3,488). happensAt(walking(br1gt_3),488).
orientation(br1gt_2,12,489). coords(br1gt_2,86,209,489).  visible(br1gt_2,489). happensAt(inactive(br1gt_2),489). orientation(br1gt_3,160,489). coords(br1gt_3,98,140,489).  visible(br1gt_3,489). happensAt(walking(br1gt_3),489).
orientation(br1gt_2,12,490). coords(br1gt_2,86,209,490).  visible(br1gt_2,490). happensAt(inactive(br1gt_2),490). orientation(br1gt_3,160,490). coords(br1gt_3,99,140,490).  visible(br1gt_3,490). happensAt(walking(br1gt_3),490).
orientation(br1gt_2,12,491). coords(br1gt_2,86,209,491).  visible(br1gt_2,491). happensAt(inactive(br1gt_2),491). orientation(br1gt_3,160,491). coords(br1gt_3,101,139,491).  visible(br1gt_3,491). happensAt(walking(br1gt_3),491).
orientation(br1gt_2,12,492). coords(br1gt_2,86,209,492).  visible(br1gt_2,492). happensAt(inactive(br1gt_2),492). orientation(br1gt_3,160,492). coords(br1gt_3,101,139,492).  visible(br1gt_3,492). happensAt(walking(br1gt_3),492).
orientation(br1gt_2,12,493). coords(br1gt_2,86,209,493).  visible(br1gt_2,493). happensAt(inactive(br1gt_2),493). orientation(br1gt_3,160,493). coords(br1gt_3,102,139,493).  visible(br1gt_3,493). happensAt(walking(br1gt_3),493).
orientation(br1gt_2,12,494). coords(br1gt_2,86,209,494).  visible(br1gt_2,494). happensAt(inactive(br1gt_2),494). orientation(br1gt_3,160,494). coords(br1gt_3,102,137,494).  visible(br1gt_3,494). happensAt(walking(br1gt_3),494).
orientation(br1gt_2,12,495). coords(br1gt_2,86,209,495).  visible(br1gt_2,495). happensAt(inactive(br1gt_2),495). orientation(br1gt_3,160,495). coords(br1gt_3,103,137,495).  visible(br1gt_3,495). happensAt(walking(br1gt_3),495).
orientation(br1gt_2,12,496). coords(br1gt_2,86,209,496).  visible(br1gt_2,496). happensAt(inactive(br1gt_2),496). orientation(br1gt_3,160,496). coords(br1gt_3,103,137,496).  visible(br1gt_3,496). happensAt(walking(br1gt_3),496).
orientation(br1gt_2,12,497). coords(br1gt_2,86,209,497).  visible(br1gt_2,497). happensAt(inactive(br1gt_2),497). orientation(br1gt_3,160,497). coords(br1gt_3,103,136,497).  visible(br1gt_3,497). happensAt(walking(br1gt_3),497).
orientation(br1gt_2,12,498). coords(br1gt_2,86,209,498).  visible(br1gt_2,498). happensAt(inactive(br1gt_2),498). orientation(br1gt_3,157,498). coords(br1gt_3,104,136,498).  visible(br1gt_3,498). happensAt(walking(br1gt_3),498).
orientation(br1gt_2,12,499). coords(br1gt_2,86,209,499).  visible(br1gt_2,499). happensAt(inactive(br1gt_2),499). orientation(br1gt_3,157,499). coords(br1gt_3,106,135,499).  visible(br1gt_3,499). happensAt(walking(br1gt_3),499).
orientation(br1gt_2,12,500). coords(br1gt_2,86,209,500).  visible(br1gt_2,500). happensAt(inactive(br1gt_2),500). orientation(br1gt_3,157,500). coords(br1gt_3,109,134,500).  visible(br1gt_3,500). happensAt(walking(br1gt_3),500).
orientation(br1gt_2,12,501). coords(br1gt_2,86,209,501).  visible(br1gt_2,501). happensAt(inactive(br1gt_2),501). orientation(br1gt_3,157,501). coords(br1gt_3,110,133,501).  visible(br1gt_3,501). happensAt(walking(br1gt_3),501).
orientation(br1gt_2,12,502). coords(br1gt_2,86,209,502).  visible(br1gt_2,502). happensAt(inactive(br1gt_2),502). orientation(br1gt_3,157,502). coords(br1gt_3,112,133,502).  visible(br1gt_3,502). happensAt(walking(br1gt_3),502).
#show holdsAt/2.