
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

	orientation(lb1gt_1,12,480). coords(lb1gt_1,206,252,480).  visible(lb1gt_1,480). happensAt(walking(lb1gt_1),480). orientation(lb1gt_2,159,480). coords(lb1gt_2,191,190,480).  visible(lb1gt_2,480). happensAt(walking(lb1gt_2),480).
orientation(lb1gt_1,12,481). coords(lb1gt_1,207,254,481).  visible(lb1gt_1,481). happensAt(walking(lb1gt_1),481). orientation(lb1gt_2,161,481). coords(lb1gt_2,191,193,481).  visible(lb1gt_2,481). happensAt(walking(lb1gt_2),481).
orientation(lb1gt_1,12,482). coords(lb1gt_1,207,256,482).  visible(lb1gt_1,482). happensAt(walking(lb1gt_1),482). orientation(lb1gt_2,162,482). coords(lb1gt_2,192,196,482).  visible(lb1gt_2,482). happensAt(walking(lb1gt_2),482).
orientation(lb1gt_1,12,483). coords(lb1gt_1,208,259,483).  visible(lb1gt_1,483). happensAt(walking(lb1gt_1),483). orientation(lb1gt_2,171,483). coords(lb1gt_2,193,197,483).  visible(lb1gt_2,483). happensAt(walking(lb1gt_2),483).
orientation(lb1gt_1,12,484). coords(lb1gt_1,208,261,484).  visible(lb1gt_1,484). happensAt(walking(lb1gt_1),484). orientation(lb1gt_2,170,484). coords(lb1gt_2,194,199,484).  visible(lb1gt_2,484). happensAt(walking(lb1gt_2),484).
orientation(lb1gt_1,12,485). coords(lb1gt_1,208,263,485).  visible(lb1gt_1,485). happensAt(walking(lb1gt_1),485). orientation(lb1gt_2,170,485). coords(lb1gt_2,195,201,485).  visible(lb1gt_2,485). happensAt(walking(lb1gt_2),485).
orientation(lb1gt_1,12,486). coords(lb1gt_1,209,264,486).  visible(lb1gt_1,486). happensAt(walking(lb1gt_1),486). orientation(lb1gt_2,172,486). coords(lb1gt_2,195,201,486).  visible(lb1gt_2,486). happensAt(walking(lb1gt_2),486).
orientation(lb1gt_1,12,487). coords(lb1gt_1,209,264,487).  visible(lb1gt_1,487). happensAt(walking(lb1gt_1),487). orientation(lb1gt_2,178,487). coords(lb1gt_2,196,202,487).  visible(lb1gt_2,487). happensAt(walking(lb1gt_2),487).
orientation(lb1gt_1,17,488). coords(lb1gt_1,209,265,488).  visible(lb1gt_1,488). happensAt(walking(lb1gt_1),488). orientation(lb1gt_2,178,488). coords(lb1gt_2,196,202,488).  visible(lb1gt_2,488). happensAt(walking(lb1gt_2),488).
orientation(lb1gt_1,21,489). coords(lb1gt_1,210,267,489).  visible(lb1gt_1,489). happensAt(walking(lb1gt_1),489). orientation(lb1gt_2,0,489). coords(lb1gt_2,197,204,489).  visible(lb1gt_2,489). happensAt(walking(lb1gt_2),489).
orientation(lb1gt_1,23,490). coords(lb1gt_1,211,268,490).  visible(lb1gt_1,490). happensAt(walking(lb1gt_1),490). orientation(lb1gt_2,1,490). coords(lb1gt_2,198,205,490).  visible(lb1gt_2,490). happensAt(walking(lb1gt_2),490).
orientation(lb1gt_1,23,491). coords(lb1gt_1,211,268,491).  visible(lb1gt_1,491). happensAt(walking(lb1gt_1),491). orientation(lb1gt_2,1,491). coords(lb1gt_2,198,207,491).  visible(lb1gt_2,491). happensAt(walking(lb1gt_2),491).
orientation(lb1gt_1,29,492). coords(lb1gt_1,212,268,492).  visible(lb1gt_1,492). happensAt(walking(lb1gt_1),492). orientation(lb1gt_2,1,492). coords(lb1gt_2,199,208,492).  visible(lb1gt_2,492). happensAt(walking(lb1gt_2),492).
orientation(lb1gt_1,31,493). coords(lb1gt_1,212,268,493).  visible(lb1gt_1,493). happensAt(walking(lb1gt_1),493). orientation(lb1gt_2,1,493). coords(lb1gt_2,199,209,493).  visible(lb1gt_2,493). happensAt(walking(lb1gt_2),493).
orientation(lb1gt_1,25,494). coords(lb1gt_1,212,269,494).  visible(lb1gt_1,494). happensAt(walking(lb1gt_1),494). orientation(lb1gt_2,1,494). coords(lb1gt_2,199,212,494).  visible(lb1gt_2,494). happensAt(walking(lb1gt_2),494).
orientation(lb1gt_1,25,495). coords(lb1gt_1,212,271,495).  visible(lb1gt_1,495). happensAt(walking(lb1gt_1),495). orientation(lb1gt_2,1,495). coords(lb1gt_2,199,214,495).  visible(lb1gt_2,495). happensAt(walking(lb1gt_2),495).
orientation(lb1gt_1,22,496). coords(lb1gt_1,212,272,496).  visible(lb1gt_1,496). happensAt(walking(lb1gt_1),496). orientation(lb1gt_2,1,496). coords(lb1gt_2,199,216,496).  visible(lb1gt_2,496). happensAt(walking(lb1gt_2),496).
orientation(lb1gt_1,22,497). coords(lb1gt_1,213,273,497).  visible(lb1gt_1,497). happensAt(walking(lb1gt_1),497). orientation(lb1gt_2,1,497). coords(lb1gt_2,199,221,497).  visible(lb1gt_2,497). happensAt(walking(lb1gt_2),497).
orientation(lb1gt_1,22,498). coords(lb1gt_1,213,275,498).  visible(lb1gt_1,498). happensAt(walking(lb1gt_1),498). orientation(lb1gt_2,1,498). coords(lb1gt_2,200,222,498).  visible(lb1gt_2,498). happensAt(walking(lb1gt_2),498).
orientation(lb1gt_1,22,499). coords(lb1gt_1,215,276,499).  visible(lb1gt_1,499). happensAt(walking(lb1gt_1),499). orientation(lb1gt_2,1,499). coords(lb1gt_2,200,224,499).  visible(lb1gt_2,499). happensAt(walking(lb1gt_2),499).
orientation(lb1gt_1,22,500). coords(lb1gt_1,217,278,500).  visible(lb1gt_1,500). happensAt(walking(lb1gt_1),500). orientation(lb1gt_2,1,500). coords(lb1gt_2,200,224,500).  visible(lb1gt_2,500). happensAt(walking(lb1gt_2),500).
orientation(lb1gt_1,16,501). coords(lb1gt_1,220,280,501).  visible(lb1gt_1,501). happensAt(walking(lb1gt_1),501). orientation(lb1gt_2,1,501). coords(lb1gt_2,200,224,501).  visible(lb1gt_2,501). happensAt(walking(lb1gt_2),501).
orientation(lb1gt_1,16,502). coords(lb1gt_1,221,281,502).  visible(lb1gt_1,502). happensAt(walking(lb1gt_1),502). orientation(lb1gt_2,1,502). coords(lb1gt_2,200,225,502).  visible(lb1gt_2,502). happensAt(walking(lb1gt_2),502).
orientation(lb1gt_1,16,503). coords(lb1gt_1,223,281,503).  visible(lb1gt_1,503). happensAt(walking(lb1gt_1),503). orientation(lb1gt_2,1,503). coords(lb1gt_2,200,225,503).  visible(lb1gt_2,503). happensAt(walking(lb1gt_2),503).
orientation(lb1gt_1,16,504). coords(lb1gt_1,224,281,504).  visible(lb1gt_1,504). happensAt(walking(lb1gt_1),504). orientation(lb1gt_2,6,504). coords(lb1gt_2,200,226,504).  visible(lb1gt_2,504). happensAt(walking(lb1gt_2),504).
#show holdsAt/2.