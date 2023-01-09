
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

	orientation(wk1gt_1,147,476). coords(wk1gt_1,81,79,476).  visible(wk1gt_1,476). happensAt(walking(wk1gt_1),476). orientation(wk1gt_4,147,476). coords(wk1gt_4,56,168,476).  visible(wk1gt_4,476). happensAt(active(wk1gt_4),476).
orientation(wk1gt_1,147,477). coords(wk1gt_1,81,78,477).  visible(wk1gt_1,477). happensAt(walking(wk1gt_1),477). orientation(wk1gt_4,147,477). coords(wk1gt_4,56,168,477).  visible(wk1gt_4,477). happensAt(active(wk1gt_4),477).
orientation(wk1gt_1,147,478). coords(wk1gt_1,81,78,478).  visible(wk1gt_1,478). happensAt(walking(wk1gt_1),478). orientation(wk1gt_4,147,478). coords(wk1gt_4,56,168,478).  visible(wk1gt_4,478). happensAt(active(wk1gt_4),478).
orientation(wk1gt_1,147,479). coords(wk1gt_1,81,78,479).  visible(wk1gt_1,479). happensAt(walking(wk1gt_1),479). orientation(wk1gt_4,147,479). coords(wk1gt_4,56,168,479).  visible(wk1gt_4,479). happensAt(active(wk1gt_4),479).
orientation(wk1gt_1,140,480). coords(wk1gt_1,81,78,480).  visible(wk1gt_1,480). happensAt(walking(wk1gt_1),480). orientation(wk1gt_4,147,480). coords(wk1gt_4,56,169,480).  visible(wk1gt_4,480). happensAt(active(wk1gt_4),480).
orientation(wk1gt_1,140,481). coords(wk1gt_1,81,78,481).  visible(wk1gt_1,481). happensAt(walking(wk1gt_1),481). orientation(wk1gt_4,147,481). coords(wk1gt_4,56,169,481).  visible(wk1gt_4,481). happensAt(active(wk1gt_4),481).
orientation(wk1gt_1,140,482). coords(wk1gt_1,81,77,482).  visible(wk1gt_1,482). happensAt(walking(wk1gt_1),482). orientation(wk1gt_4,147,482). coords(wk1gt_4,56,169,482).  visible(wk1gt_4,482). happensAt(active(wk1gt_4),482).
orientation(wk1gt_1,140,483). coords(wk1gt_1,81,76,483).  visible(wk1gt_1,483). happensAt(walking(wk1gt_1),483). orientation(wk1gt_4,147,483). coords(wk1gt_4,56,168,483).  visible(wk1gt_4,483). happensAt(active(wk1gt_4),483).
orientation(wk1gt_1,140,484). coords(wk1gt_1,81,75,484).  visible(wk1gt_1,484). happensAt(walking(wk1gt_1),484). orientation(wk1gt_4,147,484). coords(wk1gt_4,56,168,484).  visible(wk1gt_4,484). happensAt(active(wk1gt_4),484).
orientation(wk1gt_1,140,485). coords(wk1gt_1,81,75,485).  visible(wk1gt_1,485). happensAt(walking(wk1gt_1),485). orientation(wk1gt_4,147,485). coords(wk1gt_4,56,169,485).  visible(wk1gt_4,485). happensAt(active(wk1gt_4),485).
orientation(wk1gt_1,140,486). coords(wk1gt_1,81,73,486).  visible(wk1gt_1,486). happensAt(walking(wk1gt_1),486). orientation(wk1gt_4,147,486). coords(wk1gt_4,56,168,486).  visible(wk1gt_4,486). happensAt(active(wk1gt_4),486).
orientation(wk1gt_1,140,487). coords(wk1gt_1,81,72,487).  visible(wk1gt_1,487). happensAt(walking(wk1gt_1),487). orientation(wk1gt_4,153,487). coords(wk1gt_4,56,168,487).  visible(wk1gt_4,487). happensAt(active(wk1gt_4),487).
orientation(wk1gt_1,140,488). coords(wk1gt_1,81,72,488).  visible(wk1gt_1,488). happensAt(walking(wk1gt_1),488). orientation(wk1gt_4,154,488). coords(wk1gt_4,56,168,488).  visible(wk1gt_4,488). happensAt(active(wk1gt_4),488).
orientation(wk1gt_1,140,489). coords(wk1gt_1,81,72,489).  visible(wk1gt_1,489). happensAt(walking(wk1gt_1),489). orientation(wk1gt_4,154,489). coords(wk1gt_4,56,168,489).  visible(wk1gt_4,489). happensAt(active(wk1gt_4),489).
orientation(wk1gt_1,140,490). coords(wk1gt_1,81,72,490).  visible(wk1gt_1,490). happensAt(walking(wk1gt_1),490). orientation(wk1gt_4,154,490). coords(wk1gt_4,56,168,490).  visible(wk1gt_4,490). happensAt(active(wk1gt_4),490).
orientation(wk1gt_1,140,491). coords(wk1gt_1,81,72,491).  visible(wk1gt_1,491). happensAt(walking(wk1gt_1),491). orientation(wk1gt_4,157,491). coords(wk1gt_4,56,169,491).  visible(wk1gt_4,491). happensAt(active(wk1gt_4),491).
orientation(wk1gt_1,140,492). coords(wk1gt_1,81,71,492).  visible(wk1gt_1,492). happensAt(walking(wk1gt_1),492). orientation(wk1gt_4,157,492). coords(wk1gt_4,56,169,492).  visible(wk1gt_4,492). happensAt(active(wk1gt_4),492).
orientation(wk1gt_1,140,493). coords(wk1gt_1,81,70,493).  visible(wk1gt_1,493). happensAt(walking(wk1gt_1),493). orientation(wk1gt_4,157,493). coords(wk1gt_4,56,169,493).  visible(wk1gt_4,493). happensAt(active(wk1gt_4),493).
orientation(wk1gt_1,140,494). coords(wk1gt_1,81,70,494).  visible(wk1gt_1,494). happensAt(walking(wk1gt_1),494). orientation(wk1gt_4,157,494). coords(wk1gt_4,56,170,494).  visible(wk1gt_4,494). happensAt(active(wk1gt_4),494).
orientation(wk1gt_1,140,495). coords(wk1gt_1,81,69,495).  visible(wk1gt_1,495). happensAt(walking(wk1gt_1),495). orientation(wk1gt_4,157,495). coords(wk1gt_4,56,170,495).  visible(wk1gt_4,495). happensAt(active(wk1gt_4),495).
orientation(wk1gt_1,140,496). coords(wk1gt_1,80,69,496).  visible(wk1gt_1,496). happensAt(walking(wk1gt_1),496). orientation(wk1gt_4,157,496). coords(wk1gt_4,56,170,496).  visible(wk1gt_4,496). happensAt(active(wk1gt_4),496).
orientation(wk1gt_1,140,497). coords(wk1gt_1,80,68,497).  visible(wk1gt_1,497). happensAt(walking(wk1gt_1),497). orientation(wk1gt_4,157,497). coords(wk1gt_4,56,170,497).  visible(wk1gt_4,497). happensAt(active(wk1gt_4),497).
orientation(wk1gt_1,140,498). coords(wk1gt_1,80,67,498).  visible(wk1gt_1,498). happensAt(walking(wk1gt_1),498). orientation(wk1gt_4,157,498). coords(wk1gt_4,56,170,498).  visible(wk1gt_4,498). happensAt(active(wk1gt_4),498).
orientation(wk1gt_1,140,499). coords(wk1gt_1,80,67,499).  visible(wk1gt_1,499). happensAt(walking(wk1gt_1),499). orientation(wk1gt_4,157,499). coords(wk1gt_4,56,170,499).  visible(wk1gt_4,499). happensAt(active(wk1gt_4),499).
orientation(wk1gt_1,140,500). coords(wk1gt_1,80,66,500).  visible(wk1gt_1,500). happensAt(walking(wk1gt_1),500). orientation(wk1gt_4,157,500). coords(wk1gt_4,56,170,500).  visible(wk1gt_4,500). happensAt(active(wk1gt_4),500).
#show holdsAt/2.