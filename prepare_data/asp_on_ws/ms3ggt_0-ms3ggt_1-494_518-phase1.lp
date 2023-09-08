
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

	orientation(ms3ggt_0,163,494). coords(ms3ggt_0,75,186,494).  visible(ms3ggt_0,494). happensAt(walking(ms3ggt_0),494). orientation(ms3ggt_1,29,494). coords(ms3ggt_1,256,250,494).  visible(ms3ggt_1,494). happensAt(walking(ms3ggt_1),494).
orientation(ms3ggt_0,163,495). coords(ms3ggt_0,75,187,495).  visible(ms3ggt_0,495). happensAt(walking(ms3ggt_0),495). orientation(ms3ggt_1,29,495). coords(ms3ggt_1,259,252,495).  visible(ms3ggt_1,495). happensAt(walking(ms3ggt_1),495).
orientation(ms3ggt_0,163,496). coords(ms3ggt_0,75,188,496).  visible(ms3ggt_0,496). happensAt(walking(ms3ggt_0),496). orientation(ms3ggt_1,29,496). coords(ms3ggt_1,259,253,496).  visible(ms3ggt_1,496). happensAt(walking(ms3ggt_1),496).
orientation(ms3ggt_0,163,497). coords(ms3ggt_0,75,188,497).  visible(ms3ggt_0,497). happensAt(walking(ms3ggt_0),497). orientation(ms3ggt_1,29,497). coords(ms3ggt_1,259,257,497).  visible(ms3ggt_1,497). happensAt(walking(ms3ggt_1),497).
orientation(ms3ggt_0,163,498). coords(ms3ggt_0,75,189,498).  visible(ms3ggt_0,498). happensAt(walking(ms3ggt_0),498). orientation(ms3ggt_1,29,498). coords(ms3ggt_1,260,260,498).  visible(ms3ggt_1,498). happensAt(walking(ms3ggt_1),498).
orientation(ms3ggt_0,163,499). coords(ms3ggt_0,75,189,499).  visible(ms3ggt_0,499). happensAt(walking(ms3ggt_0),499). orientation(ms3ggt_1,29,499). coords(ms3ggt_1,260,263,499).  visible(ms3ggt_1,499). happensAt(walking(ms3ggt_1),499).
orientation(ms3ggt_0,163,500). coords(ms3ggt_0,75,189,500).  visible(ms3ggt_0,500). happensAt(walking(ms3ggt_0),500). orientation(ms3ggt_1,29,500). coords(ms3ggt_1,262,267,500).  visible(ms3ggt_1,500). happensAt(walking(ms3ggt_1),500).
orientation(ms3ggt_0,163,501). coords(ms3ggt_0,75,190,501).  visible(ms3ggt_0,501). happensAt(walking(ms3ggt_0),501). orientation(ms3ggt_1,29,501). coords(ms3ggt_1,263,269,501).  visible(ms3ggt_1,501). happensAt(walking(ms3ggt_1),501).
orientation(ms3ggt_0,163,502). coords(ms3ggt_0,75,190,502).  visible(ms3ggt_0,502). happensAt(walking(ms3ggt_0),502). orientation(ms3ggt_1,29,502). coords(ms3ggt_1,265,271,502).  visible(ms3ggt_1,502). happensAt(walking(ms3ggt_1),502).
orientation(ms3ggt_0,163,503). coords(ms3ggt_0,74,190,503).  visible(ms3ggt_0,503). happensAt(walking(ms3ggt_0),503). orientation(ms3ggt_1,29,503). coords(ms3ggt_1,267,271,503).  visible(ms3ggt_1,503). happensAt(walking(ms3ggt_1),503).
orientation(ms3ggt_0,163,504). coords(ms3ggt_0,74,190,504).  visible(ms3ggt_0,504). happensAt(walking(ms3ggt_0),504). orientation(ms3ggt_1,29,504). coords(ms3ggt_1,268,271,504).  visible(ms3ggt_1,504). happensAt(walking(ms3ggt_1),504).
orientation(ms3ggt_0,163,505). coords(ms3ggt_0,74,190,505).  visible(ms3ggt_0,505). happensAt(walking(ms3ggt_0),505). orientation(ms3ggt_1,29,505). coords(ms3ggt_1,268,271,505).  visible(ms3ggt_1,505). happensAt(walking(ms3ggt_1),505).
orientation(ms3ggt_0,163,506). coords(ms3ggt_0,74,191,506).  visible(ms3ggt_0,506). happensAt(walking(ms3ggt_0),506). orientation(ms3ggt_1,29,506). coords(ms3ggt_1,268,271,506).  visible(ms3ggt_1,506). happensAt(walking(ms3ggt_1),506).
orientation(ms3ggt_0,163,507). coords(ms3ggt_0,74,191,507).  visible(ms3ggt_0,507). happensAt(walking(ms3ggt_0),507). orientation(ms3ggt_1,45,507). coords(ms3ggt_1,269,271,507).  visible(ms3ggt_1,507). happensAt(walking(ms3ggt_1),507).
orientation(ms3ggt_0,163,508). coords(ms3ggt_0,74,191,508).  visible(ms3ggt_0,508). happensAt(walking(ms3ggt_0),508). orientation(ms3ggt_1,49,508). coords(ms3ggt_1,270,271,508).  visible(ms3ggt_1,508). happensAt(walking(ms3ggt_1),508).
orientation(ms3ggt_0,163,509). coords(ms3ggt_0,74,191,509).  visible(ms3ggt_0,509). happensAt(walking(ms3ggt_0),509). orientation(ms3ggt_1,52,509). coords(ms3ggt_1,271,271,509).  visible(ms3ggt_1,509). happensAt(walking(ms3ggt_1),509).
orientation(ms3ggt_0,163,510). coords(ms3ggt_0,74,191,510).  visible(ms3ggt_0,510). happensAt(walking(ms3ggt_0),510). orientation(ms3ggt_1,52,510). coords(ms3ggt_1,273,272,510).  visible(ms3ggt_1,510). happensAt(walking(ms3ggt_1),510).
orientation(ms3ggt_0,163,511). coords(ms3ggt_0,74,191,511).  visible(ms3ggt_0,511). happensAt(inactive(ms3ggt_0),511). orientation(ms3ggt_1,74,511). coords(ms3ggt_1,276,273,511).  visible(ms3ggt_1,511). happensAt(walking(ms3ggt_1),511).
orientation(ms3ggt_0,163,512). coords(ms3ggt_0,74,191,512).  visible(ms3ggt_0,512). happensAt(inactive(ms3ggt_0),512). orientation(ms3ggt_1,74,512). coords(ms3ggt_1,277,275,512).  visible(ms3ggt_1,512). happensAt(walking(ms3ggt_1),512).
orientation(ms3ggt_0,163,513). coords(ms3ggt_0,74,191,513).  visible(ms3ggt_0,513). happensAt(inactive(ms3ggt_0),513). orientation(ms3ggt_1,74,513). coords(ms3ggt_1,277,276,513).  visible(ms3ggt_1,513). happensAt(walking(ms3ggt_1),513).
orientation(ms3ggt_0,163,514). coords(ms3ggt_0,74,191,514).  visible(ms3ggt_0,514). happensAt(inactive(ms3ggt_0),514). orientation(ms3ggt_1,83,514). coords(ms3ggt_1,277,277,514).  visible(ms3ggt_1,514). happensAt(walking(ms3ggt_1),514).
orientation(ms3ggt_0,163,515). coords(ms3ggt_0,74,191,515).  visible(ms3ggt_0,515). happensAt(inactive(ms3ggt_0),515). orientation(ms3ggt_1,83,515). coords(ms3ggt_1,278,279,515).  visible(ms3ggt_1,515). happensAt(walking(ms3ggt_1),515).
orientation(ms3ggt_0,163,516). coords(ms3ggt_0,74,191,516).  visible(ms3ggt_0,516). happensAt(inactive(ms3ggt_0),516). orientation(ms3ggt_1,83,516). coords(ms3ggt_1,280,280,516).  visible(ms3ggt_1,516). happensAt(walking(ms3ggt_1),516).
orientation(ms3ggt_0,163,517). coords(ms3ggt_0,74,191,517).  visible(ms3ggt_0,517). happensAt(inactive(ms3ggt_0),517). orientation(ms3ggt_1,83,517). coords(ms3ggt_1,280,282,517).  visible(ms3ggt_1,517). happensAt(walking(ms3ggt_1),517).
orientation(ms3ggt_0,163,518). coords(ms3ggt_0,74,191,518).  visible(ms3ggt_0,518). happensAt(inactive(ms3ggt_0),518). orientation(ms3ggt_1,83,518). coords(ms3ggt_1,284,282,518).  happensAt(disappear(ms3ggt_1),518). happensAt(walking(ms3ggt_1),518).
#show holdsAt/2.