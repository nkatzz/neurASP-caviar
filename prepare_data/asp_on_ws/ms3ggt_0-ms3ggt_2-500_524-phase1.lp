
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
orientation(ms3ggt_0,163,513). coords(ms3ggt_0,74,191,513).  visible(ms3ggt_0,513). happensAt(inactive(ms3ggt_0),513). orientation(ms3ggt_2,147,513). coords(ms3ggt_2,88,116,513).  visible(ms3ggt_2,513). happensAt(walking(ms3ggt_2),513).
orientation(ms3ggt_0,163,514). coords(ms3ggt_0,74,191,514).  visible(ms3ggt_0,514). happensAt(inactive(ms3ggt_0),514). orientation(ms3ggt_2,147,514). coords(ms3ggt_2,88,116,514).  visible(ms3ggt_2,514). happensAt(walking(ms3ggt_2),514).
orientation(ms3ggt_0,163,515). coords(ms3ggt_0,74,191,515).  visible(ms3ggt_0,515). happensAt(inactive(ms3ggt_0),515). orientation(ms3ggt_2,147,515). coords(ms3ggt_2,88,115,515).  visible(ms3ggt_2,515). happensAt(walking(ms3ggt_2),515).
orientation(ms3ggt_0,163,516). coords(ms3ggt_0,74,191,516).  visible(ms3ggt_0,516). happensAt(inactive(ms3ggt_0),516). orientation(ms3ggt_2,147,516). coords(ms3ggt_2,88,115,516).  visible(ms3ggt_2,516). happensAt(walking(ms3ggt_2),516).
orientation(ms3ggt_0,163,517). coords(ms3ggt_0,74,191,517).  visible(ms3ggt_0,517). happensAt(inactive(ms3ggt_0),517). orientation(ms3ggt_2,147,517). coords(ms3ggt_2,88,115,517).  visible(ms3ggt_2,517). happensAt(walking(ms3ggt_2),517).
orientation(ms3ggt_0,163,518). coords(ms3ggt_0,74,191,518).  visible(ms3ggt_0,518). happensAt(inactive(ms3ggt_0),518). orientation(ms3ggt_2,147,518). coords(ms3ggt_2,88,114,518).  visible(ms3ggt_2,518). happensAt(walking(ms3ggt_2),518).
orientation(ms3ggt_0,163,519). coords(ms3ggt_0,74,191,519).  visible(ms3ggt_0,519). happensAt(inactive(ms3ggt_0),519). orientation(ms3ggt_2,147,519). coords(ms3ggt_2,88,113,519).  visible(ms3ggt_2,519). happensAt(walking(ms3ggt_2),519).
orientation(ms3ggt_0,163,520). coords(ms3ggt_0,74,191,520).  visible(ms3ggt_0,520). happensAt(inactive(ms3ggt_0),520). orientation(ms3ggt_2,147,520). coords(ms3ggt_2,87,112,520).  visible(ms3ggt_2,520). happensAt(walking(ms3ggt_2),520).
orientation(ms3ggt_0,163,521). coords(ms3ggt_0,74,191,521).  visible(ms3ggt_0,521). happensAt(inactive(ms3ggt_0),521). orientation(ms3ggt_2,141,521). coords(ms3ggt_2,87,112,521).  visible(ms3ggt_2,521). happensAt(walking(ms3ggt_2),521).
orientation(ms3ggt_0,163,522). coords(ms3ggt_0,74,191,522).  visible(ms3ggt_0,522). happensAt(inactive(ms3ggt_0),522). orientation(ms3ggt_2,141,522). coords(ms3ggt_2,86,110,522).  visible(ms3ggt_2,522). happensAt(walking(ms3ggt_2),522).
orientation(ms3ggt_0,163,523). coords(ms3ggt_0,74,191,523).  visible(ms3ggt_0,523). happensAt(inactive(ms3ggt_0),523). orientation(ms3ggt_2,141,523). coords(ms3ggt_2,86,110,523).  visible(ms3ggt_2,523). happensAt(walking(ms3ggt_2),523).
orientation(ms3ggt_0,168,524). coords(ms3ggt_0,73,191,524).  visible(ms3ggt_0,524). happensAt(inactive(ms3ggt_0),524). orientation(ms3ggt_2,141,524). coords(ms3ggt_2,86,109,524).  visible(ms3ggt_2,524). happensAt(walking(ms3ggt_2),524).
#show holdsAt/2.