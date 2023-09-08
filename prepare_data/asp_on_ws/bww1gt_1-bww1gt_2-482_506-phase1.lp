
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

	orientation(bww1gt_1,0,482). coords(bww1gt_1,105,264,482).  visible(bww1gt_1,482). happensAt(inactive(bww1gt_1),482). orientation(bww1gt_2,156,482). coords(bww1gt_2,172,166,482).  visible(bww1gt_2,482). happensAt(walking(bww1gt_2),482).
orientation(bww1gt_1,0,483). coords(bww1gt_1,105,264,483).  visible(bww1gt_1,483). happensAt(inactive(bww1gt_1),483). orientation(bww1gt_2,160,483). coords(bww1gt_2,174,168,483).  visible(bww1gt_2,483). happensAt(walking(bww1gt_2),483).
orientation(bww1gt_1,0,484). coords(bww1gt_1,105,264,484).  visible(bww1gt_1,484). happensAt(inactive(bww1gt_1),484). orientation(bww1gt_2,158,484). coords(bww1gt_2,175,168,484).  visible(bww1gt_2,484). happensAt(walking(bww1gt_2),484).
orientation(bww1gt_1,0,485). coords(bww1gt_1,105,264,485).  visible(bww1gt_1,485). happensAt(inactive(bww1gt_1),485). orientation(bww1gt_2,158,485). coords(bww1gt_2,176,170,485).  visible(bww1gt_2,485). happensAt(walking(bww1gt_2),485).
orientation(bww1gt_1,0,486). coords(bww1gt_1,105,264,486).  visible(bww1gt_1,486). happensAt(inactive(bww1gt_1),486). orientation(bww1gt_2,154,486). coords(bww1gt_2,177,172,486).  visible(bww1gt_2,486). happensAt(walking(bww1gt_2),486).
orientation(bww1gt_1,0,487). coords(bww1gt_1,105,264,487).  visible(bww1gt_1,487). happensAt(inactive(bww1gt_1),487). orientation(bww1gt_2,153,487). coords(bww1gt_2,178,174,487).  visible(bww1gt_2,487). happensAt(walking(bww1gt_2),487).
orientation(bww1gt_1,0,488). coords(bww1gt_1,105,264,488).  visible(bww1gt_1,488). happensAt(inactive(bww1gt_1),488). orientation(bww1gt_2,155,488). coords(bww1gt_2,178,175,488).  visible(bww1gt_2,488). happensAt(walking(bww1gt_2),488).
orientation(bww1gt_1,0,489). coords(bww1gt_1,105,264,489).  visible(bww1gt_1,489). happensAt(inactive(bww1gt_1),489). orientation(bww1gt_2,157,489). coords(bww1gt_2,179,176,489).  visible(bww1gt_2,489). happensAt(walking(bww1gt_2),489).
orientation(bww1gt_1,0,490). coords(bww1gt_1,105,264,490).  visible(bww1gt_1,490). happensAt(inactive(bww1gt_1),490). orientation(bww1gt_2,155,490). coords(bww1gt_2,179,177,490).  visible(bww1gt_2,490). happensAt(walking(bww1gt_2),490).
orientation(bww1gt_1,0,491). coords(bww1gt_1,105,264,491).  visible(bww1gt_1,491). happensAt(inactive(bww1gt_1),491). orientation(bww1gt_2,152,491). coords(bww1gt_2,179,177,491).  visible(bww1gt_2,491). happensAt(walking(bww1gt_2),491).
orientation(bww1gt_1,0,492). coords(bww1gt_1,105,264,492).  visible(bww1gt_1,492). happensAt(inactive(bww1gt_1),492). orientation(bww1gt_2,157,492). coords(bww1gt_2,179,177,492).  visible(bww1gt_2,492). happensAt(walking(bww1gt_2),492).
orientation(bww1gt_1,0,493). coords(bww1gt_1,105,264,493).  visible(bww1gt_1,493). happensAt(inactive(bww1gt_1),493). orientation(bww1gt_2,157,493). coords(bww1gt_2,179,178,493).  visible(bww1gt_2,493). happensAt(walking(bww1gt_2),493).
orientation(bww1gt_1,0,494). coords(bww1gt_1,105,264,494).  visible(bww1gt_1,494). happensAt(inactive(bww1gt_1),494). orientation(bww1gt_2,154,494). coords(bww1gt_2,179,178,494).  visible(bww1gt_2,494). happensAt(walking(bww1gt_2),494).
orientation(bww1gt_1,0,495). coords(bww1gt_1,105,264,495).  visible(bww1gt_1,495). happensAt(inactive(bww1gt_1),495). orientation(bww1gt_2,156,495). coords(bww1gt_2,180,179,495).  visible(bww1gt_2,495). happensAt(walking(bww1gt_2),495).
orientation(bww1gt_1,0,496). coords(bww1gt_1,105,264,496).  visible(bww1gt_1,496). happensAt(inactive(bww1gt_1),496). orientation(bww1gt_2,150,496). coords(bww1gt_2,180,180,496).  visible(bww1gt_2,496). happensAt(walking(bww1gt_2),496).
orientation(bww1gt_1,173,497). coords(bww1gt_1,104,263,497).  visible(bww1gt_1,497). happensAt(inactive(bww1gt_1),497). orientation(bww1gt_2,153,497). coords(bww1gt_2,180,180,497).  visible(bww1gt_2,497). happensAt(walking(bww1gt_2),497).
orientation(bww1gt_1,173,498). coords(bww1gt_1,104,263,498).  visible(bww1gt_1,498). happensAt(inactive(bww1gt_1),498). orientation(bww1gt_2,152,498). coords(bww1gt_2,181,181,498).  visible(bww1gt_2,498). happensAt(walking(bww1gt_2),498).
orientation(bww1gt_1,173,499). coords(bww1gt_1,104,263,499).  visible(bww1gt_1,499). happensAt(inactive(bww1gt_1),499). orientation(bww1gt_2,164,499). coords(bww1gt_2,181,182,499).  visible(bww1gt_2,499). happensAt(walking(bww1gt_2),499).
orientation(bww1gt_1,173,500). coords(bww1gt_1,104,263,500).  visible(bww1gt_1,500). happensAt(inactive(bww1gt_1),500). orientation(bww1gt_2,169,500). coords(bww1gt_2,181,183,500).  visible(bww1gt_2,500). happensAt(walking(bww1gt_2),500).
orientation(bww1gt_1,173,501). coords(bww1gt_1,104,263,501).  visible(bww1gt_1,501). happensAt(inactive(bww1gt_1),501). orientation(bww1gt_2,169,501). coords(bww1gt_2,181,183,501).  visible(bww1gt_2,501). happensAt(walking(bww1gt_2),501).
orientation(bww1gt_1,173,502). coords(bww1gt_1,104,263,502).  visible(bww1gt_1,502). happensAt(inactive(bww1gt_1),502). orientation(bww1gt_2,169,502). coords(bww1gt_2,181,184,502).  visible(bww1gt_2,502). happensAt(walking(bww1gt_2),502).
orientation(bww1gt_1,173,503). coords(bww1gt_1,104,263,503).  visible(bww1gt_1,503). happensAt(inactive(bww1gt_1),503). orientation(bww1gt_2,169,503). coords(bww1gt_2,181,186,503).  visible(bww1gt_2,503). happensAt(walking(bww1gt_2),503).
orientation(bww1gt_1,173,504). coords(bww1gt_1,103,263,504).  visible(bww1gt_1,504). happensAt(walking(bww1gt_1),504). orientation(bww1gt_2,169,504). coords(bww1gt_2,181,188,504).  visible(bww1gt_2,504). happensAt(walking(bww1gt_2),504).
orientation(bww1gt_1,173,505). coords(bww1gt_1,103,263,505).  visible(bww1gt_1,505). happensAt(walking(bww1gt_1),505). orientation(bww1gt_2,169,505). coords(bww1gt_2,181,189,505).  visible(bww1gt_2,505). happensAt(walking(bww1gt_2),505).
orientation(bww1gt_1,173,506). coords(bww1gt_1,103,263,506).  visible(bww1gt_1,506). happensAt(walking(bww1gt_1),506). orientation(bww1gt_2,169,506). coords(bww1gt_2,181,190,506).  visible(bww1gt_2,506). happensAt(walking(bww1gt_2),506).
#show holdsAt/2.