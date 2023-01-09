
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

	orientation(fra2gt_3,150,478). coords(fra2gt_3,38,154,478).  visible(fra2gt_3,478). happensAt(walking(fra2gt_3),478). orientation(fra2gt_4,150,478). coords(fra2gt_4,64,81,478).  visible(fra2gt_4,478). happensAt(walking(fra2gt_4),478).
orientation(fra2gt_3,144,479). coords(fra2gt_3,38,154,479).  visible(fra2gt_3,479). happensAt(walking(fra2gt_3),479). orientation(fra2gt_4,150,479). coords(fra2gt_4,64,81,479).  visible(fra2gt_4,479). happensAt(walking(fra2gt_4),479).
orientation(fra2gt_3,144,480). coords(fra2gt_3,38,153,480).  visible(fra2gt_3,480). happensAt(walking(fra2gt_3),480). orientation(fra2gt_4,150,480). coords(fra2gt_4,64,81,480).  visible(fra2gt_4,480). happensAt(walking(fra2gt_4),480).
orientation(fra2gt_3,144,481). coords(fra2gt_3,38,151,481).  visible(fra2gt_3,481). happensAt(walking(fra2gt_3),481). orientation(fra2gt_4,150,481). coords(fra2gt_4,64,81,481).  visible(fra2gt_4,481). happensAt(walking(fra2gt_4),481).
orientation(fra2gt_3,144,482). coords(fra2gt_3,38,151,482).  visible(fra2gt_3,482). happensAt(walking(fra2gt_3),482). orientation(fra2gt_4,150,482). coords(fra2gt_4,64,81,482).  visible(fra2gt_4,482). happensAt(walking(fra2gt_4),482).
orientation(fra2gt_3,144,483). coords(fra2gt_3,38,149,483).  visible(fra2gt_3,483). happensAt(walking(fra2gt_3),483). orientation(fra2gt_4,150,483). coords(fra2gt_4,64,81,483).  visible(fra2gt_4,483). happensAt(walking(fra2gt_4),483).
orientation(fra2gt_3,144,484). coords(fra2gt_3,38,148,484).  visible(fra2gt_3,484). happensAt(walking(fra2gt_3),484). orientation(fra2gt_4,150,484). coords(fra2gt_4,64,81,484).  visible(fra2gt_4,484). happensAt(walking(fra2gt_4),484).
orientation(fra2gt_3,144,485). coords(fra2gt_3,38,147,485).  visible(fra2gt_3,485). happensAt(walking(fra2gt_3),485). orientation(fra2gt_4,150,485). coords(fra2gt_4,64,81,485).  visible(fra2gt_4,485). happensAt(walking(fra2gt_4),485).
orientation(fra2gt_3,144,486). coords(fra2gt_3,38,147,486).  visible(fra2gt_3,486). happensAt(walking(fra2gt_3),486). orientation(fra2gt_4,150,486). coords(fra2gt_4,64,81,486).  visible(fra2gt_4,486). happensAt(walking(fra2gt_4),486).
orientation(fra2gt_3,144,487). coords(fra2gt_3,38,147,487).  visible(fra2gt_3,487). happensAt(walking(fra2gt_3),487). orientation(fra2gt_4,150,487). coords(fra2gt_4,64,81,487).  visible(fra2gt_4,487). happensAt(walking(fra2gt_4),487).
orientation(fra2gt_3,144,488). coords(fra2gt_3,38,146,488).  visible(fra2gt_3,488). happensAt(walking(fra2gt_3),488). orientation(fra2gt_4,150,488). coords(fra2gt_4,64,81,488).  visible(fra2gt_4,488). happensAt(walking(fra2gt_4),488).
orientation(fra2gt_3,144,489). coords(fra2gt_3,38,146,489).  visible(fra2gt_3,489). happensAt(walking(fra2gt_3),489). orientation(fra2gt_4,150,489). coords(fra2gt_4,64,81,489).  visible(fra2gt_4,489). happensAt(walking(fra2gt_4),489).
orientation(fra2gt_3,144,490). coords(fra2gt_3,39,145,490).  visible(fra2gt_3,490). happensAt(walking(fra2gt_3),490). orientation(fra2gt_4,150,490). coords(fra2gt_4,64,81,490).  visible(fra2gt_4,490). happensAt(walking(fra2gt_4),490).
orientation(fra2gt_3,144,491). coords(fra2gt_3,39,145,491).  visible(fra2gt_3,491). happensAt(walking(fra2gt_3),491). orientation(fra2gt_4,146,491). coords(fra2gt_4,63,82,491).  visible(fra2gt_4,491). happensAt(walking(fra2gt_4),491).
orientation(fra2gt_3,144,492). coords(fra2gt_3,39,142,492).  visible(fra2gt_3,492). happensAt(walking(fra2gt_3),492). orientation(fra2gt_4,146,492). coords(fra2gt_4,63,82,492).  visible(fra2gt_4,492). happensAt(walking(fra2gt_4),492).
orientation(fra2gt_3,144,493). coords(fra2gt_3,39,141,493).  visible(fra2gt_3,493). happensAt(walking(fra2gt_3),493). orientation(fra2gt_4,146,493). coords(fra2gt_4,63,82,493).  visible(fra2gt_4,493). happensAt(walking(fra2gt_4),493).
orientation(fra2gt_3,144,494). coords(fra2gt_3,39,141,494).  visible(fra2gt_3,494). happensAt(walking(fra2gt_3),494). orientation(fra2gt_4,146,494). coords(fra2gt_4,63,82,494).  visible(fra2gt_4,494). happensAt(walking(fra2gt_4),494).
orientation(fra2gt_3,144,495). coords(fra2gt_3,39,139,495).  visible(fra2gt_3,495). happensAt(walking(fra2gt_3),495). orientation(fra2gt_4,146,495). coords(fra2gt_4,63,82,495).  visible(fra2gt_4,495). happensAt(walking(fra2gt_4),495).
orientation(fra2gt_3,144,496). coords(fra2gt_3,39,138,496).  visible(fra2gt_3,496). happensAt(walking(fra2gt_3),496). orientation(fra2gt_4,146,496). coords(fra2gt_4,63,82,496).  visible(fra2gt_4,496). happensAt(walking(fra2gt_4),496).
orientation(fra2gt_3,144,497). coords(fra2gt_3,39,137,497).  visible(fra2gt_3,497). happensAt(walking(fra2gt_3),497). orientation(fra2gt_4,146,497). coords(fra2gt_4,63,82,497).  visible(fra2gt_4,497). happensAt(walking(fra2gt_4),497).
orientation(fra2gt_3,144,498). coords(fra2gt_3,39,136,498).  visible(fra2gt_3,498). happensAt(walking(fra2gt_3),498). orientation(fra2gt_4,146,498). coords(fra2gt_4,63,82,498).  visible(fra2gt_4,498). happensAt(walking(fra2gt_4),498).
orientation(fra2gt_3,144,499). coords(fra2gt_3,40,136,499).  visible(fra2gt_3,499). happensAt(walking(fra2gt_3),499). orientation(fra2gt_4,146,499). coords(fra2gt_4,63,82,499).  visible(fra2gt_4,499). happensAt(walking(fra2gt_4),499).
orientation(fra2gt_3,135,500). coords(fra2gt_3,41,135,500).  visible(fra2gt_3,500). happensAt(walking(fra2gt_3),500). orientation(fra2gt_4,146,500). coords(fra2gt_4,63,82,500).  visible(fra2gt_4,500). happensAt(walking(fra2gt_4),500).
orientation(fra2gt_3,144,501). coords(fra2gt_3,41,134,501).  visible(fra2gt_3,501). happensAt(walking(fra2gt_3),501). orientation(fra2gt_4,146,501). coords(fra2gt_4,63,82,501).  visible(fra2gt_4,501). happensAt(walking(fra2gt_4),501).
orientation(fra2gt_3,153,502). coords(fra2gt_3,41,132,502).  visible(fra2gt_3,502). happensAt(walking(fra2gt_3),502). orientation(fra2gt_4,146,502). coords(fra2gt_4,63,82,502).  visible(fra2gt_4,502). happensAt(walking(fra2gt_4),502).
#show holdsAt/2.