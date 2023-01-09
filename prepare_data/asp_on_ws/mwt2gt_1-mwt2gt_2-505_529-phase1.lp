
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

	orientation(mwt2gt_1,156,505). coords(mwt2gt_1,54,153,505).  visible(mwt2gt_1,505). happensAt(walking(mwt2gt_1),505). orientation(mwt2gt_2,151,505). coords(mwt2gt_2,48,163,505).  visible(mwt2gt_2,505). happensAt(walking(mwt2gt_2),505).
orientation(mwt2gt_1,156,506). coords(mwt2gt_1,53,153,506).  visible(mwt2gt_1,506). happensAt(walking(mwt2gt_1),506). orientation(mwt2gt_2,153,506). coords(mwt2gt_2,48,164,506).  visible(mwt2gt_2,506). happensAt(walking(mwt2gt_2),506).
orientation(mwt2gt_1,156,507). coords(mwt2gt_1,53,153,507).  visible(mwt2gt_1,507). happensAt(walking(mwt2gt_1),507). orientation(mwt2gt_2,149,507). coords(mwt2gt_2,48,164,507).  visible(mwt2gt_2,507). happensAt(walking(mwt2gt_2),507).
orientation(mwt2gt_1,156,508). coords(mwt2gt_1,51,153,508).  visible(mwt2gt_1,508). happensAt(walking(mwt2gt_1),508). orientation(mwt2gt_2,156,508). coords(mwt2gt_2,48,164,508).  visible(mwt2gt_2,508). happensAt(walking(mwt2gt_2),508).
orientation(mwt2gt_1,156,509). coords(mwt2gt_1,50,154,509).  visible(mwt2gt_1,509). happensAt(walking(mwt2gt_1),509). orientation(mwt2gt_2,152,509). coords(mwt2gt_2,48,164,509).  visible(mwt2gt_2,509). happensAt(walking(mwt2gt_2),509).
orientation(mwt2gt_1,156,510). coords(mwt2gt_1,50,154,510).  visible(mwt2gt_1,510). happensAt(walking(mwt2gt_1),510). orientation(mwt2gt_2,152,510). coords(mwt2gt_2,46,165,510).  visible(mwt2gt_2,510). happensAt(walking(mwt2gt_2),510).
orientation(mwt2gt_1,153,511). coords(mwt2gt_1,50,154,511).  visible(mwt2gt_1,511). happensAt(walking(mwt2gt_1),511). orientation(mwt2gt_2,153,511). coords(mwt2gt_2,46,165,511).  visible(mwt2gt_2,511). happensAt(walking(mwt2gt_2),511).
orientation(mwt2gt_1,153,512). coords(mwt2gt_1,50,155,512).  visible(mwt2gt_1,512). happensAt(walking(mwt2gt_1),512). orientation(mwt2gt_2,152,512). coords(mwt2gt_2,46,165,512).  visible(mwt2gt_2,512). happensAt(walking(mwt2gt_2),512).
orientation(mwt2gt_1,153,513). coords(mwt2gt_1,50,155,513).  visible(mwt2gt_1,513). happensAt(walking(mwt2gt_1),513). orientation(mwt2gt_2,152,513). coords(mwt2gt_2,46,166,513).  visible(mwt2gt_2,513). happensAt(walking(mwt2gt_2),513).
orientation(mwt2gt_1,153,514). coords(mwt2gt_1,50,155,514).  visible(mwt2gt_1,514). happensAt(walking(mwt2gt_1),514). orientation(mwt2gt_2,152,514). coords(mwt2gt_2,45,166,514).  visible(mwt2gt_2,514). happensAt(walking(mwt2gt_2),514).
orientation(mwt2gt_1,153,515). coords(mwt2gt_1,50,155,515).  visible(mwt2gt_1,515). happensAt(walking(mwt2gt_1),515). orientation(mwt2gt_2,152,515). coords(mwt2gt_2,45,166,515).  visible(mwt2gt_2,515). happensAt(walking(mwt2gt_2),515).
orientation(mwt2gt_1,153,516). coords(mwt2gt_1,50,156,516).  visible(mwt2gt_1,516). happensAt(walking(mwt2gt_1),516). orientation(mwt2gt_2,152,516). coords(mwt2gt_2,45,166,516).  visible(mwt2gt_2,516). happensAt(walking(mwt2gt_2),516).
orientation(mwt2gt_1,153,517). coords(mwt2gt_1,50,155,517).  visible(mwt2gt_1,517). happensAt(walking(mwt2gt_1),517). orientation(mwt2gt_2,152,517). coords(mwt2gt_2,44,166,517).  visible(mwt2gt_2,517). happensAt(walking(mwt2gt_2),517).
orientation(mwt2gt_1,153,518). coords(mwt2gt_1,49,156,518).  visible(mwt2gt_1,518). happensAt(walking(mwt2gt_1),518). orientation(mwt2gt_2,152,518). coords(mwt2gt_2,43,167,518).  visible(mwt2gt_2,518). happensAt(walking(mwt2gt_2),518).
orientation(mwt2gt_1,153,519). coords(mwt2gt_1,48,156,519).  visible(mwt2gt_1,519). happensAt(walking(mwt2gt_1),519). orientation(mwt2gt_2,152,519). coords(mwt2gt_2,43,167,519).  visible(mwt2gt_2,519). happensAt(walking(mwt2gt_2),519).
orientation(mwt2gt_1,153,520). coords(mwt2gt_1,48,156,520).  visible(mwt2gt_1,520). happensAt(walking(mwt2gt_1),520). orientation(mwt2gt_2,152,520). coords(mwt2gt_2,43,167,520).  visible(mwt2gt_2,520). happensAt(walking(mwt2gt_2),520).
orientation(mwt2gt_1,153,521). coords(mwt2gt_1,46,156,521).  visible(mwt2gt_1,521). happensAt(walking(mwt2gt_1),521). orientation(mwt2gt_2,152,521). coords(mwt2gt_2,43,167,521).  visible(mwt2gt_2,521). happensAt(walking(mwt2gt_2),521).
orientation(mwt2gt_1,153,522). coords(mwt2gt_1,46,156,522).  visible(mwt2gt_1,522). happensAt(walking(mwt2gt_1),522). orientation(mwt2gt_2,152,522). coords(mwt2gt_2,42,167,522).  visible(mwt2gt_2,522). happensAt(walking(mwt2gt_2),522).
orientation(mwt2gt_1,153,523). coords(mwt2gt_1,46,156,523).  visible(mwt2gt_1,523). happensAt(walking(mwt2gt_1),523). orientation(mwt2gt_2,152,523). coords(mwt2gt_2,42,167,523).  visible(mwt2gt_2,523). happensAt(walking(mwt2gt_2),523).
orientation(mwt2gt_1,153,524). coords(mwt2gt_1,46,156,524).  visible(mwt2gt_1,524). happensAt(walking(mwt2gt_1),524). orientation(mwt2gt_2,152,524). coords(mwt2gt_2,42,167,524).  visible(mwt2gt_2,524). happensAt(walking(mwt2gt_2),524).
orientation(mwt2gt_1,153,525). coords(mwt2gt_1,46,156,525).  visible(mwt2gt_1,525). happensAt(walking(mwt2gt_1),525). orientation(mwt2gt_2,152,525). coords(mwt2gt_2,42,167,525).  visible(mwt2gt_2,525). happensAt(walking(mwt2gt_2),525).
orientation(mwt2gt_1,153,526). coords(mwt2gt_1,46,156,526).  visible(mwt2gt_1,526). happensAt(walking(mwt2gt_1),526). orientation(mwt2gt_2,152,526). coords(mwt2gt_2,42,167,526).  visible(mwt2gt_2,526). happensAt(walking(mwt2gt_2),526).
orientation(mwt2gt_1,153,527). coords(mwt2gt_1,46,156,527).  visible(mwt2gt_1,527). happensAt(walking(mwt2gt_1),527). orientation(mwt2gt_2,152,527). coords(mwt2gt_2,40,168,527).  visible(mwt2gt_2,527). happensAt(walking(mwt2gt_2),527).
orientation(mwt2gt_1,153,528). coords(mwt2gt_1,46,156,528).  visible(mwt2gt_1,528). happensAt(walking(mwt2gt_1),528). orientation(mwt2gt_2,152,528). coords(mwt2gt_2,40,168,528).  visible(mwt2gt_2,528). happensAt(walking(mwt2gt_2),528).
orientation(mwt2gt_1,153,529). coords(mwt2gt_1,46,156,529).  visible(mwt2gt_1,529). happensAt(walking(mwt2gt_1),529). orientation(mwt2gt_2,152,529). coords(mwt2gt_2,39,169,529).  visible(mwt2gt_2,529). happensAt(walking(mwt2gt_2),529).
#show holdsAt/2.