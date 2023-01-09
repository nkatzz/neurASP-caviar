
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

	orientation(fra1gt_3,175,501). coords(fra1gt_3,101,204,501).  visible(fra1gt_3,501). happensAt(active(fra1gt_3),501). orientation(fra1gt_7,146,501). coords(fra1gt_7,45,97,501).  visible(fra1gt_7,501). happensAt(walking(fra1gt_7),501).
orientation(fra1gt_3,175,502). coords(fra1gt_3,101,204,502).  visible(fra1gt_3,502). happensAt(active(fra1gt_3),502). orientation(fra1gt_7,146,502). coords(fra1gt_7,45,98,502).  visible(fra1gt_7,502). happensAt(walking(fra1gt_7),502).
orientation(fra1gt_3,175,503). coords(fra1gt_3,101,204,503).  visible(fra1gt_3,503). happensAt(active(fra1gt_3),503). orientation(fra1gt_7,146,503). coords(fra1gt_7,45,98,503).  visible(fra1gt_7,503). happensAt(walking(fra1gt_7),503).
orientation(fra1gt_3,175,504). coords(fra1gt_3,101,204,504).  visible(fra1gt_3,504). happensAt(active(fra1gt_3),504). orientation(fra1gt_7,146,504). coords(fra1gt_7,45,99,504).  visible(fra1gt_7,504). happensAt(walking(fra1gt_7),504).
orientation(fra1gt_3,175,505). coords(fra1gt_3,101,204,505).  visible(fra1gt_3,505). happensAt(active(fra1gt_3),505). orientation(fra1gt_7,146,505). coords(fra1gt_7,45,99,505).  visible(fra1gt_7,505). happensAt(walking(fra1gt_7),505).
orientation(fra1gt_3,175,506). coords(fra1gt_3,101,204,506).  visible(fra1gt_3,506). happensAt(active(fra1gt_3),506). orientation(fra1gt_7,146,506). coords(fra1gt_7,44,100,506).  visible(fra1gt_7,506). happensAt(walking(fra1gt_7),506).
orientation(fra1gt_3,175,507). coords(fra1gt_3,101,204,507).  visible(fra1gt_3,507). happensAt(active(fra1gt_3),507). orientation(fra1gt_7,146,507). coords(fra1gt_7,44,100,507).  visible(fra1gt_7,507). happensAt(walking(fra1gt_7),507).
orientation(fra1gt_3,175,508). coords(fra1gt_3,101,204,508).  visible(fra1gt_3,508). happensAt(active(fra1gt_3),508). orientation(fra1gt_7,146,508). coords(fra1gt_7,44,100,508).  visible(fra1gt_7,508). happensAt(walking(fra1gt_7),508).
orientation(fra1gt_3,175,509). coords(fra1gt_3,101,204,509).  visible(fra1gt_3,509). happensAt(active(fra1gt_3),509). orientation(fra1gt_7,146,509). coords(fra1gt_7,44,100,509).  visible(fra1gt_7,509). happensAt(walking(fra1gt_7),509).
orientation(fra1gt_3,175,510). coords(fra1gt_3,101,204,510).  visible(fra1gt_3,510). happensAt(active(fra1gt_3),510). orientation(fra1gt_7,146,510). coords(fra1gt_7,44,100,510).  visible(fra1gt_7,510). happensAt(walking(fra1gt_7),510).
orientation(fra1gt_3,175,511). coords(fra1gt_3,101,204,511).  visible(fra1gt_3,511). happensAt(active(fra1gt_3),511). orientation(fra1gt_7,146,511). coords(fra1gt_7,44,100,511).  visible(fra1gt_7,511). happensAt(walking(fra1gt_7),511).
orientation(fra1gt_3,175,512). coords(fra1gt_3,101,204,512).  visible(fra1gt_3,512). happensAt(active(fra1gt_3),512). orientation(fra1gt_7,146,512). coords(fra1gt_7,43,100,512).  visible(fra1gt_7,512). happensAt(walking(fra1gt_7),512).
orientation(fra1gt_3,175,513). coords(fra1gt_3,101,204,513).  visible(fra1gt_3,513). happensAt(active(fra1gt_3),513). orientation(fra1gt_7,146,513). coords(fra1gt_7,43,100,513).  visible(fra1gt_7,513). happensAt(walking(fra1gt_7),513).
orientation(fra1gt_3,175,514). coords(fra1gt_3,101,204,514).  visible(fra1gt_3,514). happensAt(active(fra1gt_3),514). orientation(fra1gt_7,146,514). coords(fra1gt_7,43,100,514).  visible(fra1gt_7,514). happensAt(walking(fra1gt_7),514).
orientation(fra1gt_3,175,515). coords(fra1gt_3,101,204,515).  visible(fra1gt_3,515). happensAt(active(fra1gt_3),515). orientation(fra1gt_7,146,515). coords(fra1gt_7,43,100,515).  visible(fra1gt_7,515). happensAt(walking(fra1gt_7),515).
orientation(fra1gt_3,175,516). coords(fra1gt_3,101,204,516).  visible(fra1gt_3,516). happensAt(active(fra1gt_3),516). orientation(fra1gt_7,146,516). coords(fra1gt_7,42,100,516).  visible(fra1gt_7,516). happensAt(walking(fra1gt_7),516).
orientation(fra1gt_3,175,517). coords(fra1gt_3,101,204,517).  visible(fra1gt_3,517). happensAt(active(fra1gt_3),517). orientation(fra1gt_7,146,517). coords(fra1gt_7,42,100,517).  visible(fra1gt_7,517). happensAt(walking(fra1gt_7),517).
orientation(fra1gt_3,175,518). coords(fra1gt_3,101,204,518).  visible(fra1gt_3,518). happensAt(active(fra1gt_3),518). orientation(fra1gt_7,146,518). coords(fra1gt_7,42,100,518).  visible(fra1gt_7,518). happensAt(walking(fra1gt_7),518).
orientation(fra1gt_3,175,519). coords(fra1gt_3,101,204,519).  visible(fra1gt_3,519). happensAt(active(fra1gt_3),519). orientation(fra1gt_7,146,519). coords(fra1gt_7,42,100,519).  visible(fra1gt_7,519). happensAt(walking(fra1gt_7),519).
orientation(fra1gt_3,175,520). coords(fra1gt_3,101,204,520).  visible(fra1gt_3,520). happensAt(active(fra1gt_3),520). orientation(fra1gt_7,146,520). coords(fra1gt_7,42,101,520).  visible(fra1gt_7,520). happensAt(walking(fra1gt_7),520).
orientation(fra1gt_3,175,521). coords(fra1gt_3,101,204,521).  visible(fra1gt_3,521). happensAt(active(fra1gt_3),521). orientation(fra1gt_7,146,521). coords(fra1gt_7,42,101,521).  visible(fra1gt_7,521). happensAt(walking(fra1gt_7),521).
orientation(fra1gt_3,175,522). coords(fra1gt_3,101,204,522).  visible(fra1gt_3,522). happensAt(active(fra1gt_3),522). orientation(fra1gt_7,146,522). coords(fra1gt_7,42,101,522).  visible(fra1gt_7,522). happensAt(walking(fra1gt_7),522).
orientation(fra1gt_3,175,523). coords(fra1gt_3,101,204,523).  visible(fra1gt_3,523). happensAt(active(fra1gt_3),523). orientation(fra1gt_7,146,523). coords(fra1gt_7,42,101,523).  visible(fra1gt_7,523). happensAt(walking(fra1gt_7),523).
orientation(fra1gt_3,175,524). coords(fra1gt_3,101,204,524).  visible(fra1gt_3,524). happensAt(active(fra1gt_3),524). orientation(fra1gt_7,146,524). coords(fra1gt_7,42,101,524).  visible(fra1gt_7,524). happensAt(walking(fra1gt_7),524).
orientation(fra1gt_3,175,525). coords(fra1gt_3,101,204,525).  visible(fra1gt_3,525). happensAt(active(fra1gt_3),525). orientation(fra1gt_7,146,525). coords(fra1gt_7,42,101,525).  visible(fra1gt_7,525). happensAt(walking(fra1gt_7),525).
#show holdsAt/2.