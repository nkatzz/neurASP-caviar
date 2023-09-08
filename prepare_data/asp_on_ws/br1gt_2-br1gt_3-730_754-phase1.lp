
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

	orientation(br1gt_2,12,730). coords(br1gt_2,86,209,730).  visible(br1gt_2,730). happensAt(inactive(br1gt_2),730). orientation(br1gt_3,100,730). coords(br1gt_3,298,95,730).  visible(br1gt_3,730). happensAt(active(br1gt_3),730).
orientation(br1gt_2,12,731). coords(br1gt_2,86,209,731).  visible(br1gt_2,731). happensAt(inactive(br1gt_2),731). orientation(br1gt_3,91,731). coords(br1gt_3,298,95,731).  visible(br1gt_3,731). happensAt(active(br1gt_3),731).
orientation(br1gt_2,12,732). coords(br1gt_2,86,209,732).  visible(br1gt_2,732). happensAt(inactive(br1gt_2),732). orientation(br1gt_3,91,732). coords(br1gt_3,298,95,732).  visible(br1gt_3,732). happensAt(active(br1gt_3),732).
orientation(br1gt_2,12,733). coords(br1gt_2,86,209,733).  visible(br1gt_2,733). happensAt(inactive(br1gt_2),733). orientation(br1gt_3,91,733). coords(br1gt_3,299,95,733).  visible(br1gt_3,733). happensAt(active(br1gt_3),733).
orientation(br1gt_2,12,734). coords(br1gt_2,86,209,734).  visible(br1gt_2,734). happensAt(inactive(br1gt_2),734). orientation(br1gt_3,91,734). coords(br1gt_3,299,95,734).  visible(br1gt_3,734). happensAt(active(br1gt_3),734).
orientation(br1gt_2,12,735). coords(br1gt_2,86,209,735).  visible(br1gt_2,735). happensAt(inactive(br1gt_2),735). orientation(br1gt_3,79,735). coords(br1gt_3,299,95,735).  visible(br1gt_3,735). happensAt(active(br1gt_3),735).
orientation(br1gt_2,12,736). coords(br1gt_2,86,209,736).  visible(br1gt_2,736). happensAt(inactive(br1gt_2),736). orientation(br1gt_3,79,736). coords(br1gt_3,301,97,736).  visible(br1gt_3,736). happensAt(active(br1gt_3),736).
orientation(br1gt_2,12,737). coords(br1gt_2,86,209,737).  visible(br1gt_2,737). happensAt(inactive(br1gt_2),737). orientation(br1gt_3,79,737). coords(br1gt_3,302,97,737).  visible(br1gt_3,737). happensAt(active(br1gt_3),737).
orientation(br1gt_2,12,738). coords(br1gt_2,86,209,738).  visible(br1gt_2,738). happensAt(inactive(br1gt_2),738). orientation(br1gt_3,79,738). coords(br1gt_3,304,98,738).  visible(br1gt_3,738). happensAt(active(br1gt_3),738).
orientation(br1gt_2,12,739). coords(br1gt_2,86,209,739).  visible(br1gt_2,739). happensAt(inactive(br1gt_2),739). orientation(br1gt_3,79,739). coords(br1gt_3,304,98,739).  visible(br1gt_3,739). happensAt(active(br1gt_3),739).
orientation(br1gt_2,12,740). coords(br1gt_2,86,209,740).  visible(br1gt_2,740). happensAt(inactive(br1gt_2),740). orientation(br1gt_3,79,740). coords(br1gt_3,304,98,740).  visible(br1gt_3,740). happensAt(active(br1gt_3),740).
orientation(br1gt_2,12,741). coords(br1gt_2,86,209,741).  visible(br1gt_2,741). happensAt(inactive(br1gt_2),741). orientation(br1gt_3,79,741). coords(br1gt_3,304,98,741).  visible(br1gt_3,741). happensAt(active(br1gt_3),741).
orientation(br1gt_2,12,742). coords(br1gt_2,86,209,742).  visible(br1gt_2,742). happensAt(inactive(br1gt_2),742). orientation(br1gt_3,79,742). coords(br1gt_3,304,98,742).  visible(br1gt_3,742). happensAt(active(br1gt_3),742).
orientation(br1gt_2,12,743). coords(br1gt_2,86,209,743).  visible(br1gt_2,743). happensAt(inactive(br1gt_2),743). orientation(br1gt_3,83,743). coords(br1gt_3,304,98,743).  visible(br1gt_3,743). happensAt(active(br1gt_3),743).
orientation(br1gt_2,12,744). coords(br1gt_2,86,209,744).  visible(br1gt_2,744). happensAt(inactive(br1gt_2),744). orientation(br1gt_3,83,744). coords(br1gt_3,304,98,744).  visible(br1gt_3,744). happensAt(active(br1gt_3),744).
orientation(br1gt_2,12,745). coords(br1gt_2,86,209,745).  visible(br1gt_2,745). happensAt(inactive(br1gt_2),745). orientation(br1gt_3,83,745). coords(br1gt_3,304,100,745).  visible(br1gt_3,745). happensAt(active(br1gt_3),745).
orientation(br1gt_2,12,746). coords(br1gt_2,86,209,746).  visible(br1gt_2,746). happensAt(inactive(br1gt_2),746). orientation(br1gt_3,83,746). coords(br1gt_3,304,102,746).  visible(br1gt_3,746). happensAt(active(br1gt_3),746).
orientation(br1gt_2,12,747). coords(br1gt_2,86,209,747).  visible(br1gt_2,747). happensAt(inactive(br1gt_2),747). orientation(br1gt_3,83,747). coords(br1gt_3,304,103,747).  visible(br1gt_3,747). happensAt(active(br1gt_3),747).
orientation(br1gt_2,12,748). coords(br1gt_2,86,209,748).  visible(br1gt_2,748). happensAt(inactive(br1gt_2),748). orientation(br1gt_3,83,748). coords(br1gt_3,304,104,748).  visible(br1gt_3,748). happensAt(active(br1gt_3),748).
orientation(br1gt_2,12,749). coords(br1gt_2,86,209,749).  visible(br1gt_2,749). happensAt(inactive(br1gt_2),749). orientation(br1gt_3,83,749). coords(br1gt_3,304,105,749).  visible(br1gt_3,749). happensAt(active(br1gt_3),749).
orientation(br1gt_2,12,750). coords(br1gt_2,86,209,750).  visible(br1gt_2,750). happensAt(inactive(br1gt_2),750). orientation(br1gt_3,83,750). coords(br1gt_3,304,105,750).  visible(br1gt_3,750). happensAt(active(br1gt_3),750).
orientation(br1gt_2,12,751). coords(br1gt_2,86,209,751).  visible(br1gt_2,751). happensAt(inactive(br1gt_2),751). orientation(br1gt_3,83,751). coords(br1gt_3,304,105,751).  visible(br1gt_3,751). happensAt(active(br1gt_3),751).
orientation(br1gt_2,12,752). coords(br1gt_2,86,209,752).  visible(br1gt_2,752). happensAt(inactive(br1gt_2),752). orientation(br1gt_3,89,752). coords(br1gt_3,303,105,752).  visible(br1gt_3,752). happensAt(active(br1gt_3),752).
orientation(br1gt_2,12,753). coords(br1gt_2,86,209,753).  visible(br1gt_2,753). happensAt(inactive(br1gt_2),753). orientation(br1gt_3,89,753). coords(br1gt_3,303,105,753).  visible(br1gt_3,753). happensAt(active(br1gt_3),753).
orientation(br1gt_2,12,754). coords(br1gt_2,86,209,754).  visible(br1gt_2,754). happensAt(inactive(br1gt_2),754). orientation(br1gt_3,89,754). coords(br1gt_3,303,106,754).  visible(br1gt_3,754). happensAt(active(br1gt_3),754).
#show holdsAt/2.