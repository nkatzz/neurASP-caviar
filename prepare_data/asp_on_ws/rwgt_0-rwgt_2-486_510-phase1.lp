
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

	orientation(rwgt_0,137,486). coords(rwgt_0,316,283,486).  visible(rwgt_0,486). happensAt(inactive(rwgt_0),486). orientation(rwgt_2,125,486). coords(rwgt_2,168,19,486).  visible(rwgt_2,486). happensAt(walking(rwgt_2),486).
orientation(rwgt_0,137,487). coords(rwgt_0,316,283,487).  visible(rwgt_0,487). happensAt(inactive(rwgt_0),487). orientation(rwgt_2,125,487). coords(rwgt_2,168,19,487).  visible(rwgt_2,487). happensAt(walking(rwgt_2),487).
orientation(rwgt_0,137,488). coords(rwgt_0,316,283,488).  visible(rwgt_0,488). happensAt(inactive(rwgt_0),488). orientation(rwgt_2,125,488). coords(rwgt_2,168,20,488).  visible(rwgt_2,488). happensAt(walking(rwgt_2),488).
orientation(rwgt_0,137,489). coords(rwgt_0,316,283,489).  visible(rwgt_0,489). happensAt(inactive(rwgt_0),489). orientation(rwgt_2,124,489). coords(rwgt_2,167,20,489).  visible(rwgt_2,489). happensAt(walking(rwgt_2),489).
orientation(rwgt_0,137,490). coords(rwgt_0,316,283,490).  visible(rwgt_0,490). happensAt(inactive(rwgt_0),490). orientation(rwgt_2,124,490). coords(rwgt_2,167,20,490).  visible(rwgt_2,490). happensAt(walking(rwgt_2),490).
orientation(rwgt_0,137,491). coords(rwgt_0,316,283,491).  visible(rwgt_0,491). happensAt(inactive(rwgt_0),491). orientation(rwgt_2,124,491). coords(rwgt_2,167,21,491).  visible(rwgt_2,491). happensAt(walking(rwgt_2),491).
orientation(rwgt_0,137,492). coords(rwgt_0,316,283,492).  visible(rwgt_0,492). happensAt(inactive(rwgt_0),492). orientation(rwgt_2,127,492). coords(rwgt_2,167,21,492).  visible(rwgt_2,492). happensAt(walking(rwgt_2),492).
orientation(rwgt_0,137,493). coords(rwgt_0,316,283,493).  visible(rwgt_0,493). happensAt(inactive(rwgt_0),493). orientation(rwgt_2,124,493). coords(rwgt_2,167,22,493).  visible(rwgt_2,493). happensAt(walking(rwgt_2),493).
orientation(rwgt_0,137,494). coords(rwgt_0,316,283,494).  visible(rwgt_0,494). happensAt(inactive(rwgt_0),494). orientation(rwgt_2,124,494). coords(rwgt_2,167,22,494).  visible(rwgt_2,494). happensAt(walking(rwgt_2),494).
orientation(rwgt_0,137,495). coords(rwgt_0,316,283,495).  visible(rwgt_0,495). happensAt(inactive(rwgt_0),495). orientation(rwgt_2,124,495). coords(rwgt_2,167,22,495).  visible(rwgt_2,495). happensAt(walking(rwgt_2),495).
orientation(rwgt_0,137,496). coords(rwgt_0,316,283,496).  visible(rwgt_0,496). happensAt(inactive(rwgt_0),496). orientation(rwgt_2,124,496). coords(rwgt_2,166,24,496).  visible(rwgt_2,496). happensAt(walking(rwgt_2),496).
orientation(rwgt_0,137,497). coords(rwgt_0,316,283,497).  visible(rwgt_0,497). happensAt(inactive(rwgt_0),497). orientation(rwgt_2,124,497). coords(rwgt_2,166,26,497).  visible(rwgt_2,497). happensAt(walking(rwgt_2),497).
orientation(rwgt_0,137,498). coords(rwgt_0,316,283,498).  visible(rwgt_0,498). happensAt(inactive(rwgt_0),498). orientation(rwgt_2,124,498). coords(rwgt_2,166,27,498).  visible(rwgt_2,498). happensAt(walking(rwgt_2),498).
orientation(rwgt_0,137,499). coords(rwgt_0,316,283,499).  visible(rwgt_0,499). happensAt(inactive(rwgt_0),499). orientation(rwgt_2,124,499). coords(rwgt_2,166,28,499).  visible(rwgt_2,499). happensAt(walking(rwgt_2),499).
orientation(rwgt_0,137,500). coords(rwgt_0,316,283,500).  visible(rwgt_0,500). happensAt(inactive(rwgt_0),500). orientation(rwgt_2,124,500). coords(rwgt_2,165,29,500).  visible(rwgt_2,500). happensAt(walking(rwgt_2),500).
orientation(rwgt_0,137,501). coords(rwgt_0,316,283,501).  visible(rwgt_0,501). happensAt(inactive(rwgt_0),501). orientation(rwgt_2,124,501). coords(rwgt_2,165,29,501).  visible(rwgt_2,501). happensAt(walking(rwgt_2),501).
orientation(rwgt_0,137,502). coords(rwgt_0,316,283,502).  visible(rwgt_0,502). happensAt(inactive(rwgt_0),502). orientation(rwgt_2,124,502). coords(rwgt_2,165,29,502).  visible(rwgt_2,502). happensAt(walking(rwgt_2),502).
orientation(rwgt_0,137,503). coords(rwgt_0,316,283,503).  visible(rwgt_0,503). happensAt(inactive(rwgt_0),503). orientation(rwgt_2,124,503). coords(rwgt_2,165,29,503).  visible(rwgt_2,503). happensAt(walking(rwgt_2),503).
orientation(rwgt_0,137,504). coords(rwgt_0,316,283,504).  visible(rwgt_0,504). happensAt(inactive(rwgt_0),504). orientation(rwgt_2,124,504). coords(rwgt_2,164,30,504).  visible(rwgt_2,504). happensAt(walking(rwgt_2),504).
orientation(rwgt_0,137,505). coords(rwgt_0,316,283,505).  visible(rwgt_0,505). happensAt(inactive(rwgt_0),505). orientation(rwgt_2,124,505). coords(rwgt_2,164,30,505).  visible(rwgt_2,505). happensAt(walking(rwgt_2),505).
orientation(rwgt_0,137,506). coords(rwgt_0,316,283,506).  visible(rwgt_0,506). happensAt(inactive(rwgt_0),506). orientation(rwgt_2,124,506). coords(rwgt_2,164,30,506).  visible(rwgt_2,506). happensAt(walking(rwgt_2),506).
orientation(rwgt_0,137,507). coords(rwgt_0,316,283,507).  visible(rwgt_0,507). happensAt(inactive(rwgt_0),507). orientation(rwgt_2,124,507). coords(rwgt_2,164,30,507).  visible(rwgt_2,507). happensAt(walking(rwgt_2),507).
orientation(rwgt_0,137,508). coords(rwgt_0,316,283,508).  visible(rwgt_0,508). happensAt(inactive(rwgt_0),508). orientation(rwgt_2,124,508). coords(rwgt_2,164,30,508).  visible(rwgt_2,508). happensAt(walking(rwgt_2),508).
orientation(rwgt_0,137,509). coords(rwgt_0,316,283,509).  visible(rwgt_0,509). happensAt(inactive(rwgt_0),509). orientation(rwgt_2,124,509). coords(rwgt_2,163,30,509).  visible(rwgt_2,509). happensAt(walking(rwgt_2),509).
orientation(rwgt_0,137,510). coords(rwgt_0,316,283,510).  visible(rwgt_0,510). happensAt(inactive(rwgt_0),510). orientation(rwgt_2,124,510). coords(rwgt_2,163,30,510).  visible(rwgt_2,510). happensAt(walking(rwgt_2),510).
#show holdsAt/2.