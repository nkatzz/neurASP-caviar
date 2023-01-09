
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

	orientation(bww1gt_1,173,506). coords(bww1gt_1,103,263,506).  visible(bww1gt_1,506). happensAt(walking(bww1gt_1),506). orientation(bww1gt_2,169,506). coords(bww1gt_2,181,190,506).  visible(bww1gt_2,506). happensAt(walking(bww1gt_2),506).
orientation(bww1gt_1,173,507). coords(bww1gt_1,103,263,507).  visible(bww1gt_1,507). happensAt(walking(bww1gt_1),507). orientation(bww1gt_2,169,507). coords(bww1gt_2,181,192,507).  visible(bww1gt_2,507). happensAt(walking(bww1gt_2),507).
orientation(bww1gt_1,173,508). coords(bww1gt_1,103,263,508).  visible(bww1gt_1,508). happensAt(walking(bww1gt_1),508). orientation(bww1gt_2,169,508). coords(bww1gt_2,181,192,508).  visible(bww1gt_2,508). happensAt(walking(bww1gt_2),508).
orientation(bww1gt_1,173,509). coords(bww1gt_1,103,263,509).  visible(bww1gt_1,509). happensAt(walking(bww1gt_1),509). orientation(bww1gt_2,169,509). coords(bww1gt_2,181,193,509).  visible(bww1gt_2,509). happensAt(walking(bww1gt_2),509).
orientation(bww1gt_1,173,510). coords(bww1gt_1,102,263,510).  visible(bww1gt_1,510). happensAt(walking(bww1gt_1),510). orientation(bww1gt_2,166,510). coords(bww1gt_2,181,193,510).  visible(bww1gt_2,510). happensAt(walking(bww1gt_2),510).
orientation(bww1gt_1,173,511). coords(bww1gt_1,102,263,511).  visible(bww1gt_1,511). happensAt(walking(bww1gt_1),511). orientation(bww1gt_2,168,511). coords(bww1gt_2,181,194,511).  visible(bww1gt_2,511). happensAt(walking(bww1gt_2),511).
orientation(bww1gt_1,173,512). coords(bww1gt_1,102,263,512).  visible(bww1gt_1,512). happensAt(walking(bww1gt_1),512). orientation(bww1gt_2,168,512). coords(bww1gt_2,181,195,512).  visible(bww1gt_2,512). happensAt(walking(bww1gt_2),512).
orientation(bww1gt_1,173,513). coords(bww1gt_1,102,263,513).  visible(bww1gt_1,513). happensAt(walking(bww1gt_1),513). orientation(bww1gt_2,169,513). coords(bww1gt_2,181,195,513).  visible(bww1gt_2,513). happensAt(walking(bww1gt_2),513).
orientation(bww1gt_1,173,514). coords(bww1gt_1,102,263,514).  visible(bww1gt_1,514). happensAt(walking(bww1gt_1),514). orientation(bww1gt_2,169,514). coords(bww1gt_2,181,196,514).  visible(bww1gt_2,514). happensAt(walking(bww1gt_2),514).
orientation(bww1gt_1,173,515). coords(bww1gt_1,102,262,515).  visible(bww1gt_1,515). happensAt(walking(bww1gt_1),515). orientation(bww1gt_2,166,515). coords(bww1gt_2,181,196,515).  visible(bww1gt_2,515). happensAt(walking(bww1gt_2),515).
orientation(bww1gt_1,173,516). coords(bww1gt_1,102,262,516).  visible(bww1gt_1,516). happensAt(walking(bww1gt_1),516). orientation(bww1gt_2,169,516). coords(bww1gt_2,180,196,516).  visible(bww1gt_2,516). happensAt(walking(bww1gt_2),516).
orientation(bww1gt_1,173,517). coords(bww1gt_1,101,261,517).  visible(bww1gt_1,517). happensAt(walking(bww1gt_1),517). orientation(bww1gt_2,169,517). coords(bww1gt_2,180,197,517).  visible(bww1gt_2,517). happensAt(walking(bww1gt_2),517).
orientation(bww1gt_1,173,518). coords(bww1gt_1,101,261,518).  visible(bww1gt_1,518). happensAt(walking(bww1gt_1),518). orientation(bww1gt_2,169,518). coords(bww1gt_2,180,197,518).  visible(bww1gt_2,518). happensAt(walking(bww1gt_2),518).
orientation(bww1gt_1,173,519). coords(bww1gt_1,101,261,519).  visible(bww1gt_1,519). happensAt(walking(bww1gt_1),519). orientation(bww1gt_2,169,519). coords(bww1gt_2,180,197,519).  visible(bww1gt_2,519). happensAt(walking(bww1gt_2),519).
orientation(bww1gt_1,173,520). coords(bww1gt_1,101,260,520).  visible(bww1gt_1,520). happensAt(walking(bww1gt_1),520). orientation(bww1gt_2,169,520). coords(bww1gt_2,180,198,520).  visible(bww1gt_2,520). happensAt(walking(bww1gt_2),520).
orientation(bww1gt_1,173,521). coords(bww1gt_1,101,260,521).  visible(bww1gt_1,521). happensAt(walking(bww1gt_1),521). orientation(bww1gt_2,169,521). coords(bww1gt_2,179,198,521).  visible(bww1gt_2,521). happensAt(walking(bww1gt_2),521).
orientation(bww1gt_1,173,522). coords(bww1gt_1,101,259,522).  visible(bww1gt_1,522). happensAt(walking(bww1gt_1),522). orientation(bww1gt_2,169,522). coords(bww1gt_2,179,199,522).  visible(bww1gt_2,522). happensAt(walking(bww1gt_2),522).
orientation(bww1gt_1,173,523). coords(bww1gt_1,101,259,523).  visible(bww1gt_1,523). happensAt(walking(bww1gt_1),523). orientation(bww1gt_2,169,523). coords(bww1gt_2,179,201,523).  visible(bww1gt_2,523). happensAt(walking(bww1gt_2),523).
orientation(bww1gt_1,173,524). coords(bww1gt_1,101,258,524).  visible(bww1gt_1,524). happensAt(walking(bww1gt_1),524). orientation(bww1gt_2,169,524). coords(bww1gt_2,177,202,524).  visible(bww1gt_2,524). happensAt(walking(bww1gt_2),524).
orientation(bww1gt_1,173,525). coords(bww1gt_1,101,258,525).  visible(bww1gt_1,525). happensAt(walking(bww1gt_1),525). orientation(bww1gt_2,169,525). coords(bww1gt_2,177,204,525).  visible(bww1gt_2,525). happensAt(walking(bww1gt_2),525).
orientation(bww1gt_1,173,526). coords(bww1gt_1,100,257,526).  visible(bww1gt_1,526). happensAt(walking(bww1gt_1),526). orientation(bww1gt_2,169,526). coords(bww1gt_2,177,205,526).  visible(bww1gt_2,526). happensAt(walking(bww1gt_2),526).
orientation(bww1gt_1,173,527). coords(bww1gt_1,100,257,527).  visible(bww1gt_1,527). happensAt(walking(bww1gt_1),527). orientation(bww1gt_2,169,527). coords(bww1gt_2,176,206,527).  visible(bww1gt_2,527). happensAt(walking(bww1gt_2),527).
orientation(bww1gt_1,173,528). coords(bww1gt_1,100,257,528).  visible(bww1gt_1,528). happensAt(walking(bww1gt_1),528). orientation(bww1gt_2,169,528). coords(bww1gt_2,175,207,528).  visible(bww1gt_2,528). happensAt(walking(bww1gt_2),528).
orientation(bww1gt_1,173,529). coords(bww1gt_1,100,257,529).  visible(bww1gt_1,529). happensAt(walking(bww1gt_1),529). orientation(bww1gt_2,169,529). coords(bww1gt_2,175,207,529).  visible(bww1gt_2,529). happensAt(walking(bww1gt_2),529).
orientation(bww1gt_1,173,530). coords(bww1gt_1,100,257,530).  visible(bww1gt_1,530). happensAt(walking(bww1gt_1),530). orientation(bww1gt_2,169,530). coords(bww1gt_2,175,207,530).  visible(bww1gt_2,530). happensAt(walking(bww1gt_2),530).
#show holdsAt/2.