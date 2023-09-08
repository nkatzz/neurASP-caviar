
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

	orientation(wk2gt_1,122,499). coords(wk2gt_1,115,53,499).  visible(wk2gt_1,499). happensAt(inactive(wk2gt_1),499). orientation(wk2gt_4,146,499). coords(wk2gt_4,67,97,499).  visible(wk2gt_4,499). happensAt(walking(wk2gt_4),499).
orientation(wk2gt_1,122,500). coords(wk2gt_1,115,53,500).  visible(wk2gt_1,500). happensAt(inactive(wk2gt_1),500). orientation(wk2gt_4,146,500). coords(wk2gt_4,66,96,500).  visible(wk2gt_4,500). happensAt(walking(wk2gt_4),500).
orientation(wk2gt_1,122,501). coords(wk2gt_1,115,53,501).  visible(wk2gt_1,501). happensAt(inactive(wk2gt_1),501). orientation(wk2gt_4,146,501). coords(wk2gt_4,65,96,501).  visible(wk2gt_4,501). happensAt(walking(wk2gt_4),501).
orientation(wk2gt_1,122,502). coords(wk2gt_1,115,53,502).  visible(wk2gt_1,502). happensAt(inactive(wk2gt_1),502). orientation(wk2gt_4,146,502). coords(wk2gt_4,65,96,502).  visible(wk2gt_4,502). happensAt(walking(wk2gt_4),502).
orientation(wk2gt_1,122,503). coords(wk2gt_1,115,53,503).  visible(wk2gt_1,503). happensAt(inactive(wk2gt_1),503). orientation(wk2gt_4,146,503). coords(wk2gt_4,64,96,503).  visible(wk2gt_4,503). happensAt(walking(wk2gt_4),503).
orientation(wk2gt_1,122,504). coords(wk2gt_1,115,53,504).  visible(wk2gt_1,504). happensAt(inactive(wk2gt_1),504). orientation(wk2gt_4,149,504). coords(wk2gt_4,64,96,504).  visible(wk2gt_4,504). happensAt(walking(wk2gt_4),504).
orientation(wk2gt_1,122,505). coords(wk2gt_1,115,53,505).  visible(wk2gt_1,505). happensAt(inactive(wk2gt_1),505). orientation(wk2gt_4,149,505). coords(wk2gt_4,64,96,505).  visible(wk2gt_4,505). happensAt(walking(wk2gt_4),505).
orientation(wk2gt_1,122,506). coords(wk2gt_1,115,53,506).  visible(wk2gt_1,506). happensAt(inactive(wk2gt_1),506). orientation(wk2gt_4,149,506). coords(wk2gt_4,64,96,506).  visible(wk2gt_4,506). happensAt(walking(wk2gt_4),506).
orientation(wk2gt_1,122,507). coords(wk2gt_1,115,53,507).  visible(wk2gt_1,507). happensAt(inactive(wk2gt_1),507). orientation(wk2gt_4,149,507). coords(wk2gt_4,64,96,507).  visible(wk2gt_4,507). happensAt(walking(wk2gt_4),507).
orientation(wk2gt_1,122,508). coords(wk2gt_1,115,53,508).  visible(wk2gt_1,508). happensAt(inactive(wk2gt_1),508). orientation(wk2gt_4,149,508). coords(wk2gt_4,64,96,508).  visible(wk2gt_4,508). happensAt(walking(wk2gt_4),508).
orientation(wk2gt_1,122,509). coords(wk2gt_1,115,53,509).  visible(wk2gt_1,509). happensAt(inactive(wk2gt_1),509). orientation(wk2gt_4,149,509). coords(wk2gt_4,63,96,509).  visible(wk2gt_4,509). happensAt(walking(wk2gt_4),509).
orientation(wk2gt_1,122,510). coords(wk2gt_1,115,53,510).  visible(wk2gt_1,510). happensAt(inactive(wk2gt_1),510). orientation(wk2gt_4,149,510). coords(wk2gt_4,63,96,510).  visible(wk2gt_4,510). happensAt(walking(wk2gt_4),510).
orientation(wk2gt_1,122,511). coords(wk2gt_1,115,53,511).  visible(wk2gt_1,511). happensAt(inactive(wk2gt_1),511). orientation(wk2gt_4,149,511). coords(wk2gt_4,63,96,511).  visible(wk2gt_4,511). happensAt(walking(wk2gt_4),511).
orientation(wk2gt_1,122,512). coords(wk2gt_1,115,53,512).  visible(wk2gt_1,512). happensAt(inactive(wk2gt_1),512). orientation(wk2gt_4,149,512). coords(wk2gt_4,62,96,512).  visible(wk2gt_4,512). happensAt(walking(wk2gt_4),512).
orientation(wk2gt_1,122,513). coords(wk2gt_1,115,53,513).  visible(wk2gt_1,513). happensAt(inactive(wk2gt_1),513). orientation(wk2gt_4,149,513). coords(wk2gt_4,61,96,513).  visible(wk2gt_4,513). happensAt(walking(wk2gt_4),513).
orientation(wk2gt_1,122,514). coords(wk2gt_1,115,53,514).  visible(wk2gt_1,514). happensAt(inactive(wk2gt_1),514). orientation(wk2gt_4,149,514). coords(wk2gt_4,60,96,514).  visible(wk2gt_4,514). happensAt(walking(wk2gt_4),514).
orientation(wk2gt_1,122,515). coords(wk2gt_1,115,53,515).  visible(wk2gt_1,515). happensAt(inactive(wk2gt_1),515). orientation(wk2gt_4,149,515). coords(wk2gt_4,60,96,515).  visible(wk2gt_4,515). happensAt(walking(wk2gt_4),515).
orientation(wk2gt_1,122,516). coords(wk2gt_1,115,53,516).  visible(wk2gt_1,516). happensAt(inactive(wk2gt_1),516). orientation(wk2gt_4,149,516). coords(wk2gt_4,59,96,516).  visible(wk2gt_4,516). happensAt(walking(wk2gt_4),516).
orientation(wk2gt_1,122,517). coords(wk2gt_1,115,53,517).  visible(wk2gt_1,517). happensAt(inactive(wk2gt_1),517). orientation(wk2gt_4,149,517). coords(wk2gt_4,59,96,517).  visible(wk2gt_4,517). happensAt(walking(wk2gt_4),517).
orientation(wk2gt_1,122,518). coords(wk2gt_1,115,53,518).  visible(wk2gt_1,518). happensAt(inactive(wk2gt_1),518). orientation(wk2gt_4,149,518). coords(wk2gt_4,59,97,518).  visible(wk2gt_4,518). happensAt(walking(wk2gt_4),518).
orientation(wk2gt_1,122,519). coords(wk2gt_1,115,53,519).  visible(wk2gt_1,519). happensAt(inactive(wk2gt_1),519). orientation(wk2gt_4,150,519). coords(wk2gt_4,59,97,519).  visible(wk2gt_4,519). happensAt(walking(wk2gt_4),519).
orientation(wk2gt_1,122,520). coords(wk2gt_1,115,53,520).  visible(wk2gt_1,520). happensAt(inactive(wk2gt_1),520). orientation(wk2gt_4,150,520). coords(wk2gt_4,58,97,520).  visible(wk2gt_4,520). happensAt(walking(wk2gt_4),520).
orientation(wk2gt_1,122,521). coords(wk2gt_1,115,53,521).  visible(wk2gt_1,521). happensAt(inactive(wk2gt_1),521). orientation(wk2gt_4,150,521). coords(wk2gt_4,58,97,521).  visible(wk2gt_4,521). happensAt(walking(wk2gt_4),521).
orientation(wk2gt_1,122,522). coords(wk2gt_1,115,53,522).  visible(wk2gt_1,522). happensAt(inactive(wk2gt_1),522). orientation(wk2gt_4,150,522). coords(wk2gt_4,58,97,522).  visible(wk2gt_4,522). happensAt(walking(wk2gt_4),522).
orientation(wk2gt_1,122,523). coords(wk2gt_1,115,53,523).  visible(wk2gt_1,523). happensAt(inactive(wk2gt_1),523). orientation(wk2gt_4,150,523). coords(wk2gt_4,58,97,523).  visible(wk2gt_4,523). happensAt(walking(wk2gt_4),523).
#show holdsAt/2.