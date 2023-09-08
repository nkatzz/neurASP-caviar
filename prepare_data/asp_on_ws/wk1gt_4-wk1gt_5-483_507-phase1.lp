
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

	orientation(wk1gt_4,147,483). coords(wk1gt_4,56,168,483).  visible(wk1gt_4,483). happensAt(active(wk1gt_4),483). orientation(wk1gt_5,162,483). coords(wk1gt_5,57,152,483).  visible(wk1gt_5,483). happensAt(active(wk1gt_5),483).
orientation(wk1gt_4,147,484). coords(wk1gt_4,56,168,484).  visible(wk1gt_4,484). happensAt(active(wk1gt_4),484). orientation(wk1gt_5,162,484). coords(wk1gt_5,57,152,484).  visible(wk1gt_5,484). happensAt(active(wk1gt_5),484).
orientation(wk1gt_4,147,485). coords(wk1gt_4,56,169,485).  visible(wk1gt_4,485). happensAt(active(wk1gt_4),485). orientation(wk1gt_5,162,485). coords(wk1gt_5,57,152,485).  visible(wk1gt_5,485). happensAt(active(wk1gt_5),485).
orientation(wk1gt_4,147,486). coords(wk1gt_4,56,168,486).  visible(wk1gt_4,486). happensAt(active(wk1gt_4),486). orientation(wk1gt_5,162,486). coords(wk1gt_5,57,152,486).  visible(wk1gt_5,486). happensAt(active(wk1gt_5),486).
orientation(wk1gt_4,153,487). coords(wk1gt_4,56,168,487).  visible(wk1gt_4,487). happensAt(active(wk1gt_4),487). orientation(wk1gt_5,162,487). coords(wk1gt_5,57,152,487).  visible(wk1gt_5,487). happensAt(active(wk1gt_5),487).
orientation(wk1gt_4,154,488). coords(wk1gt_4,56,168,488).  visible(wk1gt_4,488). happensAt(active(wk1gt_4),488). orientation(wk1gt_5,162,488). coords(wk1gt_5,57,152,488).  visible(wk1gt_5,488). happensAt(active(wk1gt_5),488).
orientation(wk1gt_4,154,489). coords(wk1gt_4,56,168,489).  visible(wk1gt_4,489). happensAt(active(wk1gt_4),489). orientation(wk1gt_5,162,489). coords(wk1gt_5,57,151,489).  visible(wk1gt_5,489). happensAt(active(wk1gt_5),489).
orientation(wk1gt_4,154,490). coords(wk1gt_4,56,168,490).  visible(wk1gt_4,490). happensAt(active(wk1gt_4),490). orientation(wk1gt_5,162,490). coords(wk1gt_5,57,150,490).  visible(wk1gt_5,490). happensAt(active(wk1gt_5),490).
orientation(wk1gt_4,157,491). coords(wk1gt_4,56,169,491).  visible(wk1gt_4,491). happensAt(active(wk1gt_4),491). orientation(wk1gt_5,162,491). coords(wk1gt_5,57,150,491).  visible(wk1gt_5,491). happensAt(active(wk1gt_5),491).
orientation(wk1gt_4,157,492). coords(wk1gt_4,56,169,492).  visible(wk1gt_4,492). happensAt(active(wk1gt_4),492). orientation(wk1gt_5,162,492). coords(wk1gt_5,57,150,492).  visible(wk1gt_5,492). happensAt(active(wk1gt_5),492).
orientation(wk1gt_4,157,493). coords(wk1gt_4,56,169,493).  visible(wk1gt_4,493). happensAt(active(wk1gt_4),493). orientation(wk1gt_5,162,493). coords(wk1gt_5,57,150,493).  visible(wk1gt_5,493). happensAt(active(wk1gt_5),493).
orientation(wk1gt_4,157,494). coords(wk1gt_4,56,170,494).  visible(wk1gt_4,494). happensAt(active(wk1gt_4),494). orientation(wk1gt_5,162,494). coords(wk1gt_5,57,150,494).  visible(wk1gt_5,494). happensAt(active(wk1gt_5),494).
orientation(wk1gt_4,157,495). coords(wk1gt_4,56,170,495).  visible(wk1gt_4,495). happensAt(active(wk1gt_4),495). orientation(wk1gt_5,162,495). coords(wk1gt_5,57,150,495).  visible(wk1gt_5,495). happensAt(active(wk1gt_5),495).
orientation(wk1gt_4,157,496). coords(wk1gt_4,56,170,496).  visible(wk1gt_4,496). happensAt(active(wk1gt_4),496). orientation(wk1gt_5,162,496). coords(wk1gt_5,57,150,496).  visible(wk1gt_5,496). happensAt(active(wk1gt_5),496).
orientation(wk1gt_4,157,497). coords(wk1gt_4,56,170,497).  visible(wk1gt_4,497). happensAt(active(wk1gt_4),497). orientation(wk1gt_5,162,497). coords(wk1gt_5,57,150,497).  visible(wk1gt_5,497). happensAt(active(wk1gt_5),497).
orientation(wk1gt_4,157,498). coords(wk1gt_4,56,170,498).  visible(wk1gt_4,498). happensAt(active(wk1gt_4),498). orientation(wk1gt_5,162,498). coords(wk1gt_5,57,150,498).  visible(wk1gt_5,498). happensAt(inactive(wk1gt_5),498).
orientation(wk1gt_4,157,499). coords(wk1gt_4,56,170,499).  visible(wk1gt_4,499). happensAt(active(wk1gt_4),499). orientation(wk1gt_5,162,499). coords(wk1gt_5,57,150,499).  visible(wk1gt_5,499). happensAt(inactive(wk1gt_5),499).
orientation(wk1gt_4,157,500). coords(wk1gt_4,56,170,500).  visible(wk1gt_4,500). happensAt(active(wk1gt_4),500). orientation(wk1gt_5,162,500). coords(wk1gt_5,57,150,500).  visible(wk1gt_5,500). happensAt(inactive(wk1gt_5),500).
orientation(wk1gt_4,157,501). coords(wk1gt_4,56,170,501).  visible(wk1gt_4,501). happensAt(active(wk1gt_4),501). orientation(wk1gt_5,162,501). coords(wk1gt_5,57,150,501).  visible(wk1gt_5,501). happensAt(inactive(wk1gt_5),501).
orientation(wk1gt_4,157,502). coords(wk1gt_4,56,170,502).  visible(wk1gt_4,502). happensAt(active(wk1gt_4),502). orientation(wk1gt_5,162,502). coords(wk1gt_5,57,150,502).  visible(wk1gt_5,502). happensAt(inactive(wk1gt_5),502).
orientation(wk1gt_4,157,503). coords(wk1gt_4,56,170,503).  visible(wk1gt_4,503). happensAt(active(wk1gt_4),503). orientation(wk1gt_5,162,503). coords(wk1gt_5,57,150,503).  visible(wk1gt_5,503). happensAt(inactive(wk1gt_5),503).
orientation(wk1gt_4,157,504). coords(wk1gt_4,56,170,504).  visible(wk1gt_4,504). happensAt(active(wk1gt_4),504). orientation(wk1gt_5,162,504). coords(wk1gt_5,57,150,504).  visible(wk1gt_5,504). happensAt(inactive(wk1gt_5),504).
orientation(wk1gt_4,157,505). coords(wk1gt_4,56,170,505).  visible(wk1gt_4,505). happensAt(active(wk1gt_4),505). orientation(wk1gt_5,162,505). coords(wk1gt_5,57,150,505).  visible(wk1gt_5,505). happensAt(inactive(wk1gt_5),505).
orientation(wk1gt_4,157,506). coords(wk1gt_4,56,170,506).  visible(wk1gt_4,506). happensAt(active(wk1gt_4),506). orientation(wk1gt_5,162,506). coords(wk1gt_5,57,150,506).  visible(wk1gt_5,506). happensAt(inactive(wk1gt_5),506).
orientation(wk1gt_4,157,507). coords(wk1gt_4,56,170,507).  visible(wk1gt_4,507). happensAt(active(wk1gt_4),507). orientation(wk1gt_5,162,507). coords(wk1gt_5,57,150,507).  visible(wk1gt_5,507). happensAt(inactive(wk1gt_5),507).
#show holdsAt/2.