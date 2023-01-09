
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

	orientation(ms3ggt_0,160,488). coords(ms3ggt_0,76,183,488).  visible(ms3ggt_0,488). happensAt(walking(ms3ggt_0),488). orientation(ms3ggt_2,150,488). coords(ms3ggt_2,94,133,488).  visible(ms3ggt_2,488). happensAt(walking(ms3ggt_2),488).
orientation(ms3ggt_0,160,489). coords(ms3ggt_0,76,184,489).  visible(ms3ggt_0,489). happensAt(walking(ms3ggt_0),489). orientation(ms3ggt_2,150,489). coords(ms3ggt_2,94,132,489).  visible(ms3ggt_2,489). happensAt(walking(ms3ggt_2),489).
orientation(ms3ggt_0,160,490). coords(ms3ggt_0,76,184,490).  visible(ms3ggt_0,490). happensAt(walking(ms3ggt_0),490). orientation(ms3ggt_2,150,490). coords(ms3ggt_2,94,132,490).  visible(ms3ggt_2,490). happensAt(walking(ms3ggt_2),490).
orientation(ms3ggt_0,163,491). coords(ms3ggt_0,75,185,491).  visible(ms3ggt_0,491). happensAt(walking(ms3ggt_0),491). orientation(ms3ggt_2,150,491). coords(ms3ggt_2,94,131,491).  visible(ms3ggt_2,491). happensAt(walking(ms3ggt_2),491).
orientation(ms3ggt_0,163,492). coords(ms3ggt_0,75,185,492).  visible(ms3ggt_0,492). happensAt(walking(ms3ggt_0),492). orientation(ms3ggt_2,143,492). coords(ms3ggt_2,94,131,492).  visible(ms3ggt_2,492). happensAt(walking(ms3ggt_2),492).
orientation(ms3ggt_0,163,493). coords(ms3ggt_0,75,186,493).  visible(ms3ggt_0,493). happensAt(walking(ms3ggt_0),493). orientation(ms3ggt_2,143,493). coords(ms3ggt_2,94,129,493).  visible(ms3ggt_2,493). happensAt(walking(ms3ggt_2),493).
orientation(ms3ggt_0,163,494). coords(ms3ggt_0,75,186,494).  visible(ms3ggt_0,494). happensAt(walking(ms3ggt_0),494). orientation(ms3ggt_2,143,494). coords(ms3ggt_2,93,127,494).  visible(ms3ggt_2,494). happensAt(walking(ms3ggt_2),494).
orientation(ms3ggt_0,163,495). coords(ms3ggt_0,75,187,495).  visible(ms3ggt_0,495). happensAt(walking(ms3ggt_0),495). orientation(ms3ggt_2,143,495). coords(ms3ggt_2,93,126,495).  visible(ms3ggt_2,495). happensAt(walking(ms3ggt_2),495).
orientation(ms3ggt_0,163,496). coords(ms3ggt_0,75,188,496).  visible(ms3ggt_0,496). happensAt(walking(ms3ggt_0),496). orientation(ms3ggt_2,143,496). coords(ms3ggt_2,93,125,496).  visible(ms3ggt_2,496). happensAt(walking(ms3ggt_2),496).
orientation(ms3ggt_0,163,497). coords(ms3ggt_0,75,188,497).  visible(ms3ggt_0,497). happensAt(walking(ms3ggt_0),497). orientation(ms3ggt_2,143,497). coords(ms3ggt_2,93,124,497).  visible(ms3ggt_2,497). happensAt(walking(ms3ggt_2),497).
orientation(ms3ggt_0,163,498). coords(ms3ggt_0,75,189,498).  visible(ms3ggt_0,498). happensAt(walking(ms3ggt_0),498). orientation(ms3ggt_2,143,498). coords(ms3ggt_2,93,123,498).  visible(ms3ggt_2,498). happensAt(walking(ms3ggt_2),498).
orientation(ms3ggt_0,163,499). coords(ms3ggt_0,75,189,499).  visible(ms3ggt_0,499). happensAt(walking(ms3ggt_0),499). orientation(ms3ggt_2,143,499). coords(ms3ggt_2,93,122,499).  visible(ms3ggt_2,499). happensAt(walking(ms3ggt_2),499).
orientation(ms3ggt_0,163,500). coords(ms3ggt_0,75,189,500).  visible(ms3ggt_0,500). happensAt(walking(ms3ggt_0),500). orientation(ms3ggt_2,147,500). coords(ms3ggt_2,93,120,500).  visible(ms3ggt_2,500). happensAt(walking(ms3ggt_2),500).
orientation(ms3ggt_0,163,501). coords(ms3ggt_0,75,190,501).  visible(ms3ggt_0,501). happensAt(walking(ms3ggt_0),501). orientation(ms3ggt_2,147,501). coords(ms3ggt_2,92,120,501).  visible(ms3ggt_2,501). happensAt(walking(ms3ggt_2),501).
orientation(ms3ggt_0,163,502). coords(ms3ggt_0,75,190,502).  visible(ms3ggt_0,502). happensAt(walking(ms3ggt_0),502). orientation(ms3ggt_2,147,502). coords(ms3ggt_2,92,120,502).  visible(ms3ggt_2,502). happensAt(walking(ms3ggt_2),502).
orientation(ms3ggt_0,163,503). coords(ms3ggt_0,74,190,503).  visible(ms3ggt_0,503). happensAt(walking(ms3ggt_0),503). orientation(ms3ggt_2,147,503). coords(ms3ggt_2,92,120,503).  visible(ms3ggt_2,503). happensAt(walking(ms3ggt_2),503).
orientation(ms3ggt_0,163,504). coords(ms3ggt_0,74,190,504).  visible(ms3ggt_0,504). happensAt(walking(ms3ggt_0),504). orientation(ms3ggt_2,147,504). coords(ms3ggt_2,92,119,504).  visible(ms3ggt_2,504). happensAt(walking(ms3ggt_2),504).
orientation(ms3ggt_0,163,505). coords(ms3ggt_0,74,190,505).  visible(ms3ggt_0,505). happensAt(walking(ms3ggt_0),505). orientation(ms3ggt_2,147,505). coords(ms3ggt_2,92,119,505).  visible(ms3ggt_2,505). happensAt(walking(ms3ggt_2),505).
orientation(ms3ggt_0,163,506). coords(ms3ggt_0,74,191,506).  visible(ms3ggt_0,506). happensAt(walking(ms3ggt_0),506). orientation(ms3ggt_2,147,506). coords(ms3ggt_2,91,118,506).  visible(ms3ggt_2,506). happensAt(walking(ms3ggt_2),506).
orientation(ms3ggt_0,163,507). coords(ms3ggt_0,74,191,507).  visible(ms3ggt_0,507). happensAt(walking(ms3ggt_0),507). orientation(ms3ggt_2,147,507). coords(ms3ggt_2,90,117,507).  visible(ms3ggt_2,507). happensAt(walking(ms3ggt_2),507).
orientation(ms3ggt_0,163,508). coords(ms3ggt_0,74,191,508).  visible(ms3ggt_0,508). happensAt(walking(ms3ggt_0),508). orientation(ms3ggt_2,147,508). coords(ms3ggt_2,90,117,508).  visible(ms3ggt_2,508). happensAt(walking(ms3ggt_2),508).
orientation(ms3ggt_0,163,509). coords(ms3ggt_0,74,191,509).  visible(ms3ggt_0,509). happensAt(walking(ms3ggt_0),509). orientation(ms3ggt_2,147,509). coords(ms3ggt_2,90,117,509).  visible(ms3ggt_2,509). happensAt(walking(ms3ggt_2),509).
orientation(ms3ggt_0,163,510). coords(ms3ggt_0,74,191,510).  visible(ms3ggt_0,510). happensAt(walking(ms3ggt_0),510). orientation(ms3ggt_2,147,510). coords(ms3ggt_2,89,117,510).  visible(ms3ggt_2,510). happensAt(walking(ms3ggt_2),510).
orientation(ms3ggt_0,163,511). coords(ms3ggt_0,74,191,511).  visible(ms3ggt_0,511). happensAt(inactive(ms3ggt_0),511). orientation(ms3ggt_2,147,511). coords(ms3ggt_2,89,117,511).  visible(ms3ggt_2,511). happensAt(walking(ms3ggt_2),511).
orientation(ms3ggt_0,163,512). coords(ms3ggt_0,74,191,512).  visible(ms3ggt_0,512). happensAt(inactive(ms3ggt_0),512). orientation(ms3ggt_2,147,512). coords(ms3ggt_2,88,116,512).  visible(ms3ggt_2,512). happensAt(walking(ms3ggt_2),512).
#show holdsAt/2.