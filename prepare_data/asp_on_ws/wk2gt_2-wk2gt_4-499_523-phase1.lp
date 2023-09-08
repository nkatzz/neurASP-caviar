
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

	orientation(wk2gt_2,141,499). coords(wk2gt_2,99,63,499).  visible(wk2gt_2,499). happensAt(inactive(wk2gt_2),499). orientation(wk2gt_4,146,499). coords(wk2gt_4,67,97,499).  visible(wk2gt_4,499). happensAt(walking(wk2gt_4),499).
orientation(wk2gt_2,141,500). coords(wk2gt_2,99,63,500).  visible(wk2gt_2,500). happensAt(inactive(wk2gt_2),500). orientation(wk2gt_4,146,500). coords(wk2gt_4,66,96,500).  visible(wk2gt_4,500). happensAt(walking(wk2gt_4),500).
orientation(wk2gt_2,141,501). coords(wk2gt_2,99,63,501).  visible(wk2gt_2,501). happensAt(inactive(wk2gt_2),501). orientation(wk2gt_4,146,501). coords(wk2gt_4,65,96,501).  visible(wk2gt_4,501). happensAt(walking(wk2gt_4),501).
orientation(wk2gt_2,141,502). coords(wk2gt_2,99,63,502).  visible(wk2gt_2,502). happensAt(inactive(wk2gt_2),502). orientation(wk2gt_4,146,502). coords(wk2gt_4,65,96,502).  visible(wk2gt_4,502). happensAt(walking(wk2gt_4),502).
orientation(wk2gt_2,141,503). coords(wk2gt_2,99,63,503).  visible(wk2gt_2,503). happensAt(inactive(wk2gt_2),503). orientation(wk2gt_4,146,503). coords(wk2gt_4,64,96,503).  visible(wk2gt_4,503). happensAt(walking(wk2gt_4),503).
orientation(wk2gt_2,141,504). coords(wk2gt_2,99,63,504).  visible(wk2gt_2,504). happensAt(inactive(wk2gt_2),504). orientation(wk2gt_4,149,504). coords(wk2gt_4,64,96,504).  visible(wk2gt_4,504). happensAt(walking(wk2gt_4),504).
orientation(wk2gt_2,141,505). coords(wk2gt_2,99,63,505).  visible(wk2gt_2,505). happensAt(inactive(wk2gt_2),505). orientation(wk2gt_4,149,505). coords(wk2gt_4,64,96,505).  visible(wk2gt_4,505). happensAt(walking(wk2gt_4),505).
orientation(wk2gt_2,141,506). coords(wk2gt_2,99,63,506).  visible(wk2gt_2,506). happensAt(inactive(wk2gt_2),506). orientation(wk2gt_4,149,506). coords(wk2gt_4,64,96,506).  visible(wk2gt_4,506). happensAt(walking(wk2gt_4),506).
orientation(wk2gt_2,141,507). coords(wk2gt_2,99,63,507).  visible(wk2gt_2,507). happensAt(inactive(wk2gt_2),507). orientation(wk2gt_4,149,507). coords(wk2gt_4,64,96,507).  visible(wk2gt_4,507). happensAt(walking(wk2gt_4),507).
orientation(wk2gt_2,141,508). coords(wk2gt_2,99,63,508).  visible(wk2gt_2,508). happensAt(inactive(wk2gt_2),508). orientation(wk2gt_4,149,508). coords(wk2gt_4,64,96,508).  visible(wk2gt_4,508). happensAt(walking(wk2gt_4),508).
orientation(wk2gt_2,141,509). coords(wk2gt_2,99,63,509).  visible(wk2gt_2,509). happensAt(inactive(wk2gt_2),509). orientation(wk2gt_4,149,509). coords(wk2gt_4,63,96,509).  visible(wk2gt_4,509). happensAt(walking(wk2gt_4),509).
orientation(wk2gt_2,141,510). coords(wk2gt_2,99,63,510).  visible(wk2gt_2,510). happensAt(inactive(wk2gt_2),510). orientation(wk2gt_4,149,510). coords(wk2gt_4,63,96,510).  visible(wk2gt_4,510). happensAt(walking(wk2gt_4),510).
orientation(wk2gt_2,141,511). coords(wk2gt_2,99,63,511).  visible(wk2gt_2,511). happensAt(inactive(wk2gt_2),511). orientation(wk2gt_4,149,511). coords(wk2gt_4,63,96,511).  visible(wk2gt_4,511). happensAt(walking(wk2gt_4),511).
orientation(wk2gt_2,141,512). coords(wk2gt_2,99,63,512).  visible(wk2gt_2,512). happensAt(inactive(wk2gt_2),512). orientation(wk2gt_4,149,512). coords(wk2gt_4,62,96,512).  visible(wk2gt_4,512). happensAt(walking(wk2gt_4),512).
orientation(wk2gt_2,141,513). coords(wk2gt_2,99,63,513).  visible(wk2gt_2,513). happensAt(inactive(wk2gt_2),513). orientation(wk2gt_4,149,513). coords(wk2gt_4,61,96,513).  visible(wk2gt_4,513). happensAt(walking(wk2gt_4),513).
orientation(wk2gt_2,141,514). coords(wk2gt_2,99,63,514).  visible(wk2gt_2,514). happensAt(inactive(wk2gt_2),514). orientation(wk2gt_4,149,514). coords(wk2gt_4,60,96,514).  visible(wk2gt_4,514). happensAt(walking(wk2gt_4),514).
orientation(wk2gt_2,141,515). coords(wk2gt_2,99,63,515).  visible(wk2gt_2,515). happensAt(inactive(wk2gt_2),515). orientation(wk2gt_4,149,515). coords(wk2gt_4,60,96,515).  visible(wk2gt_4,515). happensAt(walking(wk2gt_4),515).
orientation(wk2gt_2,141,516). coords(wk2gt_2,99,63,516).  visible(wk2gt_2,516). happensAt(inactive(wk2gt_2),516). orientation(wk2gt_4,149,516). coords(wk2gt_4,59,96,516).  visible(wk2gt_4,516). happensAt(walking(wk2gt_4),516).
orientation(wk2gt_2,141,517). coords(wk2gt_2,99,63,517).  visible(wk2gt_2,517). happensAt(inactive(wk2gt_2),517). orientation(wk2gt_4,149,517). coords(wk2gt_4,59,96,517).  visible(wk2gt_4,517). happensAt(walking(wk2gt_4),517).
orientation(wk2gt_2,141,518). coords(wk2gt_2,99,63,518).  visible(wk2gt_2,518). happensAt(inactive(wk2gt_2),518). orientation(wk2gt_4,149,518). coords(wk2gt_4,59,97,518).  visible(wk2gt_4,518). happensAt(walking(wk2gt_4),518).
orientation(wk2gt_2,141,519). coords(wk2gt_2,99,63,519).  visible(wk2gt_2,519). happensAt(inactive(wk2gt_2),519). orientation(wk2gt_4,150,519). coords(wk2gt_4,59,97,519).  visible(wk2gt_4,519). happensAt(walking(wk2gt_4),519).
orientation(wk2gt_2,141,520). coords(wk2gt_2,99,63,520).  visible(wk2gt_2,520). happensAt(inactive(wk2gt_2),520). orientation(wk2gt_4,150,520). coords(wk2gt_4,58,97,520).  visible(wk2gt_4,520). happensAt(walking(wk2gt_4),520).
orientation(wk2gt_2,141,521). coords(wk2gt_2,99,63,521).  visible(wk2gt_2,521). happensAt(inactive(wk2gt_2),521). orientation(wk2gt_4,150,521). coords(wk2gt_4,58,97,521).  visible(wk2gt_4,521). happensAt(walking(wk2gt_4),521).
orientation(wk2gt_2,141,522). coords(wk2gt_2,99,63,522).  visible(wk2gt_2,522). happensAt(inactive(wk2gt_2),522). orientation(wk2gt_4,150,522). coords(wk2gt_4,58,97,522).  visible(wk2gt_4,522). happensAt(walking(wk2gt_4),522).
orientation(wk2gt_2,141,523). coords(wk2gt_2,99,63,523).  visible(wk2gt_2,523). happensAt(inactive(wk2gt_2),523). orientation(wk2gt_4,150,523). coords(wk2gt_4,58,97,523).  visible(wk2gt_4,523). happensAt(walking(wk2gt_4),523).
#show holdsAt/2.