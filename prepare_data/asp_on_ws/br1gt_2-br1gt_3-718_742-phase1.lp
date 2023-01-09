
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

	orientation(br1gt_2,12,718). coords(br1gt_2,86,209,718).  visible(br1gt_2,718). happensAt(inactive(br1gt_2),718). orientation(br1gt_3,107,718). coords(br1gt_3,298,95,718).  visible(br1gt_3,718). happensAt(active(br1gt_3),718).
orientation(br1gt_2,12,719). coords(br1gt_2,86,209,719).  visible(br1gt_2,719). happensAt(inactive(br1gt_2),719). orientation(br1gt_3,107,719). coords(br1gt_3,298,95,719).  visible(br1gt_3,719). happensAt(active(br1gt_3),719).
orientation(br1gt_2,12,720). coords(br1gt_2,86,209,720).  visible(br1gt_2,720). happensAt(inactive(br1gt_2),720). orientation(br1gt_3,107,720). coords(br1gt_3,298,95,720).  visible(br1gt_3,720). happensAt(active(br1gt_3),720).
orientation(br1gt_2,12,721). coords(br1gt_2,86,209,721).  visible(br1gt_2,721). happensAt(inactive(br1gt_2),721). orientation(br1gt_3,107,721). coords(br1gt_3,298,95,721).  visible(br1gt_3,721). happensAt(active(br1gt_3),721).
orientation(br1gt_2,12,722). coords(br1gt_2,86,209,722).  visible(br1gt_2,722). happensAt(inactive(br1gt_2),722). orientation(br1gt_3,107,722). coords(br1gt_3,298,95,722).  visible(br1gt_3,722). happensAt(active(br1gt_3),722).
orientation(br1gt_2,12,723). coords(br1gt_2,86,209,723).  visible(br1gt_2,723). happensAt(inactive(br1gt_2),723). orientation(br1gt_3,107,723). coords(br1gt_3,299,95,723).  visible(br1gt_3,723). happensAt(active(br1gt_3),723).
orientation(br1gt_2,12,724). coords(br1gt_2,86,209,724).  visible(br1gt_2,724). happensAt(inactive(br1gt_2),724). orientation(br1gt_3,107,724). coords(br1gt_3,298,95,724).  visible(br1gt_3,724). happensAt(active(br1gt_3),724).
orientation(br1gt_2,12,725). coords(br1gt_2,86,209,725).  visible(br1gt_2,725). happensAt(inactive(br1gt_2),725). orientation(br1gt_3,107,725). coords(br1gt_3,298,95,725).  visible(br1gt_3,725). happensAt(active(br1gt_3),725).
orientation(br1gt_2,12,726). coords(br1gt_2,86,209,726).  visible(br1gt_2,726). happensAt(inactive(br1gt_2),726). orientation(br1gt_3,107,726). coords(br1gt_3,298,95,726).  visible(br1gt_3,726). happensAt(active(br1gt_3),726).
orientation(br1gt_2,12,727). coords(br1gt_2,86,209,727).  visible(br1gt_2,727). happensAt(inactive(br1gt_2),727). orientation(br1gt_3,107,727). coords(br1gt_3,298,95,727).  visible(br1gt_3,727). happensAt(active(br1gt_3),727).
orientation(br1gt_2,12,728). coords(br1gt_2,86,209,728).  visible(br1gt_2,728). happensAt(inactive(br1gt_2),728). orientation(br1gt_3,100,728). coords(br1gt_3,298,95,728).  visible(br1gt_3,728). happensAt(active(br1gt_3),728).
orientation(br1gt_2,12,729). coords(br1gt_2,86,209,729).  visible(br1gt_2,729). happensAt(inactive(br1gt_2),729). orientation(br1gt_3,100,729). coords(br1gt_3,298,95,729).  visible(br1gt_3,729). happensAt(active(br1gt_3),729).
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
#show holdsAt/2.