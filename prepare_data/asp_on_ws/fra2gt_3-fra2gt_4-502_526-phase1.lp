
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

	orientation(fra2gt_3,153,502). coords(fra2gt_3,41,132,502).  visible(fra2gt_3,502). happensAt(walking(fra2gt_3),502). orientation(fra2gt_4,146,502). coords(fra2gt_4,63,82,502).  visible(fra2gt_4,502). happensAt(walking(fra2gt_4),502).
orientation(fra2gt_3,153,503). coords(fra2gt_3,42,132,503).  visible(fra2gt_3,503). happensAt(walking(fra2gt_3),503). orientation(fra2gt_4,146,503). coords(fra2gt_4,63,82,503).  visible(fra2gt_4,503). happensAt(walking(fra2gt_4),503).
orientation(fra2gt_3,153,504). coords(fra2gt_3,42,132,504).  visible(fra2gt_3,504). happensAt(walking(fra2gt_3),504). orientation(fra2gt_4,146,504). coords(fra2gt_4,63,82,504).  visible(fra2gt_4,504). happensAt(walking(fra2gt_4),504).
orientation(fra2gt_3,153,505). coords(fra2gt_3,42,131,505).  visible(fra2gt_3,505). happensAt(walking(fra2gt_3),505). orientation(fra2gt_4,146,505). coords(fra2gt_4,63,82,505).  visible(fra2gt_4,505). happensAt(walking(fra2gt_4),505).
orientation(fra2gt_3,144,506). coords(fra2gt_3,42,131,506).  visible(fra2gt_3,506). happensAt(walking(fra2gt_3),506). orientation(fra2gt_4,146,506). coords(fra2gt_4,63,82,506).  visible(fra2gt_4,506). happensAt(walking(fra2gt_4),506).
orientation(fra2gt_3,142,507). coords(fra2gt_3,42,131,507).  visible(fra2gt_3,507). happensAt(walking(fra2gt_3),507). orientation(fra2gt_4,146,507). coords(fra2gt_4,63,82,507).  visible(fra2gt_4,507). happensAt(walking(fra2gt_4),507).
orientation(fra2gt_3,144,508). coords(fra2gt_3,42,129,508).  visible(fra2gt_3,508). happensAt(walking(fra2gt_3),508). orientation(fra2gt_4,146,508). coords(fra2gt_4,63,82,508).  visible(fra2gt_4,508). happensAt(walking(fra2gt_4),508).
orientation(fra2gt_3,143,509). coords(fra2gt_3,42,129,509).  visible(fra2gt_3,509). happensAt(walking(fra2gt_3),509). orientation(fra2gt_4,146,509). coords(fra2gt_4,63,82,509).  visible(fra2gt_4,509). happensAt(walking(fra2gt_4),509).
orientation(fra2gt_3,141,510). coords(fra2gt_3,42,129,510).  visible(fra2gt_3,510). happensAt(walking(fra2gt_3),510). orientation(fra2gt_4,146,510). coords(fra2gt_4,63,82,510).  visible(fra2gt_4,510). happensAt(walking(fra2gt_4),510).
orientation(fra2gt_3,139,511). coords(fra2gt_3,42,128,511).  visible(fra2gt_3,511). happensAt(walking(fra2gt_3),511). orientation(fra2gt_4,146,511). coords(fra2gt_4,63,82,511).  visible(fra2gt_4,511). happensAt(walking(fra2gt_4),511).
orientation(fra2gt_3,143,512). coords(fra2gt_3,42,127,512).  visible(fra2gt_3,512). happensAt(walking(fra2gt_3),512). orientation(fra2gt_4,146,512). coords(fra2gt_4,63,82,512).  visible(fra2gt_4,512). happensAt(walking(fra2gt_4),512).
orientation(fra2gt_3,139,513). coords(fra2gt_3,43,127,513).  visible(fra2gt_3,513). happensAt(walking(fra2gt_3),513). orientation(fra2gt_4,146,513). coords(fra2gt_4,63,82,513).  visible(fra2gt_4,513). happensAt(walking(fra2gt_4),513).
orientation(fra2gt_3,139,514). coords(fra2gt_3,44,126,514).  visible(fra2gt_3,514). happensAt(walking(fra2gt_3),514). orientation(fra2gt_4,146,514). coords(fra2gt_4,63,82,514).  visible(fra2gt_4,514). happensAt(walking(fra2gt_4),514).
orientation(fra2gt_3,146,515). coords(fra2gt_3,44,125,515).  visible(fra2gt_3,515). happensAt(walking(fra2gt_3),515). orientation(fra2gt_4,146,515). coords(fra2gt_4,63,82,515).  visible(fra2gt_4,515). happensAt(walking(fra2gt_4),515).
orientation(fra2gt_3,146,516). coords(fra2gt_3,44,125,516).  visible(fra2gt_3,516). happensAt(walking(fra2gt_3),516). orientation(fra2gt_4,146,516). coords(fra2gt_4,63,82,516).  visible(fra2gt_4,516). happensAt(walking(fra2gt_4),516).
orientation(fra2gt_3,146,517). coords(fra2gt_3,44,124,517).  visible(fra2gt_3,517). happensAt(walking(fra2gt_3),517). orientation(fra2gt_4,146,517). coords(fra2gt_4,63,83,517).  visible(fra2gt_4,517). happensAt(walking(fra2gt_4),517).
orientation(fra2gt_3,146,518). coords(fra2gt_3,44,122,518).  visible(fra2gt_3,518). happensAt(walking(fra2gt_3),518). orientation(fra2gt_4,146,518). coords(fra2gt_4,63,84,518).  visible(fra2gt_4,518). happensAt(walking(fra2gt_4),518).
orientation(fra2gt_3,146,519). coords(fra2gt_3,44,122,519).  visible(fra2gt_3,519). happensAt(walking(fra2gt_3),519). orientation(fra2gt_4,146,519). coords(fra2gt_4,63,85,519).  visible(fra2gt_4,519). happensAt(walking(fra2gt_4),519).
orientation(fra2gt_3,146,520). coords(fra2gt_3,44,120,520).  visible(fra2gt_3,520). happensAt(walking(fra2gt_3),520). orientation(fra2gt_4,146,520). coords(fra2gt_4,63,87,520).  visible(fra2gt_4,520). happensAt(walking(fra2gt_4),520).
orientation(fra2gt_3,146,521). coords(fra2gt_3,44,119,521).  visible(fra2gt_3,521). happensAt(walking(fra2gt_3),521). orientation(fra2gt_4,146,521). coords(fra2gt_4,63,87,521).  visible(fra2gt_4,521). happensAt(walking(fra2gt_4),521).
orientation(fra2gt_3,146,522). coords(fra2gt_3,44,119,522).  visible(fra2gt_3,522). happensAt(walking(fra2gt_3),522). orientation(fra2gt_4,146,522). coords(fra2gt_4,63,87,522).  visible(fra2gt_4,522). happensAt(walking(fra2gt_4),522).
orientation(fra2gt_3,146,523). coords(fra2gt_3,44,118,523).  visible(fra2gt_3,523). happensAt(walking(fra2gt_3),523). orientation(fra2gt_4,146,523). coords(fra2gt_4,64,88,523).  visible(fra2gt_4,523). happensAt(walking(fra2gt_4),523).
orientation(fra2gt_3,146,524). coords(fra2gt_3,44,117,524).  visible(fra2gt_3,524). happensAt(walking(fra2gt_3),524). orientation(fra2gt_4,146,524). coords(fra2gt_4,64,88,524).  visible(fra2gt_4,524). happensAt(walking(fra2gt_4),524).
orientation(fra2gt_3,146,525). coords(fra2gt_3,44,116,525).  visible(fra2gt_3,525). happensAt(walking(fra2gt_3),525). orientation(fra2gt_4,146,525). coords(fra2gt_4,64,89,525).  visible(fra2gt_4,525). happensAt(walking(fra2gt_4),525).
orientation(fra2gt_3,146,526). coords(fra2gt_3,44,116,526).  visible(fra2gt_3,526). happensAt(walking(fra2gt_3),526). orientation(fra2gt_4,146,526). coords(fra2gt_4,64,89,526).  visible(fra2gt_4,526). happensAt(walking(fra2gt_4),526).
#show holdsAt/2.