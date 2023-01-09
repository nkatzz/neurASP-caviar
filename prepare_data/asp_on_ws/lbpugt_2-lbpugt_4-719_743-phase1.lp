
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

	orientation(lbpugt_2,127,719). coords(lbpugt_2,286,91,719).  visible(lbpugt_2,719). happensAt(walking(lbpugt_2),719). orientation(lbpugt_4,135,719). coords(lbpugt_4,219,34,719).  visible(lbpugt_4,719). happensAt(inactive(lbpugt_4),719).
orientation(lbpugt_2,127,720). coords(lbpugt_2,287,91,720).  visible(lbpugt_2,720). happensAt(walking(lbpugt_2),720). orientation(lbpugt_4,135,720). coords(lbpugt_4,219,34,720).  visible(lbpugt_4,720). happensAt(inactive(lbpugt_4),720).
orientation(lbpugt_2,127,721). coords(lbpugt_2,288,91,721).  visible(lbpugt_2,721). happensAt(walking(lbpugt_2),721). orientation(lbpugt_4,135,721). coords(lbpugt_4,219,34,721).  visible(lbpugt_4,721). happensAt(inactive(lbpugt_4),721).
orientation(lbpugt_2,124,722). coords(lbpugt_2,289,91,722).  visible(lbpugt_2,722). happensAt(walking(lbpugt_2),722). orientation(lbpugt_4,135,722). coords(lbpugt_4,219,34,722).  visible(lbpugt_4,722). happensAt(inactive(lbpugt_4),722).
orientation(lbpugt_2,121,723). coords(lbpugt_2,289,91,723).  visible(lbpugt_2,723). happensAt(walking(lbpugt_2),723). orientation(lbpugt_4,135,723). coords(lbpugt_4,219,34,723).  visible(lbpugt_4,723). happensAt(inactive(lbpugt_4),723).
orientation(lbpugt_2,121,724). coords(lbpugt_2,290,91,724).  visible(lbpugt_2,724). happensAt(walking(lbpugt_2),724). orientation(lbpugt_4,135,724). coords(lbpugt_4,219,34,724).  visible(lbpugt_4,724). happensAt(inactive(lbpugt_4),724).
orientation(lbpugt_2,116,725). coords(lbpugt_2,290,91,725).  visible(lbpugt_2,725). happensAt(walking(lbpugt_2),725). orientation(lbpugt_4,135,725). coords(lbpugt_4,219,34,725).  visible(lbpugt_4,725). happensAt(inactive(lbpugt_4),725).
orientation(lbpugt_2,113,726). coords(lbpugt_2,290,92,726).  visible(lbpugt_2,726). happensAt(walking(lbpugt_2),726). orientation(lbpugt_4,135,726). coords(lbpugt_4,219,34,726).  visible(lbpugt_4,726). happensAt(inactive(lbpugt_4),726).
orientation(lbpugt_2,108,727). coords(lbpugt_2,291,93,727).  visible(lbpugt_2,727). happensAt(walking(lbpugt_2),727). orientation(lbpugt_4,135,727). coords(lbpugt_4,219,34,727).  visible(lbpugt_4,727). happensAt(inactive(lbpugt_4),727).
orientation(lbpugt_2,102,728). coords(lbpugt_2,291,94,728).  visible(lbpugt_2,728). happensAt(walking(lbpugt_2),728). orientation(lbpugt_4,135,728). coords(lbpugt_4,219,34,728).  visible(lbpugt_4,728). happensAt(inactive(lbpugt_4),728).
orientation(lbpugt_2,104,729). coords(lbpugt_2,293,96,729).  visible(lbpugt_2,729). happensAt(walking(lbpugt_2),729). orientation(lbpugt_4,135,729). coords(lbpugt_4,219,34,729).  visible(lbpugt_4,729). happensAt(inactive(lbpugt_4),729).
orientation(lbpugt_2,102,730). coords(lbpugt_2,293,96,730).  visible(lbpugt_2,730). happensAt(walking(lbpugt_2),730). orientation(lbpugt_4,135,730). coords(lbpugt_4,219,34,730).  visible(lbpugt_4,730). happensAt(inactive(lbpugt_4),730).
orientation(lbpugt_2,102,731). coords(lbpugt_2,293,97,731).  visible(lbpugt_2,731). happensAt(walking(lbpugt_2),731). orientation(lbpugt_4,135,731). coords(lbpugt_4,219,34,731).  visible(lbpugt_4,731). happensAt(inactive(lbpugt_4),731).
orientation(lbpugt_2,99,732). coords(lbpugt_2,294,97,732).  visible(lbpugt_2,732). happensAt(walking(lbpugt_2),732). orientation(lbpugt_4,135,732). coords(lbpugt_4,219,34,732).  visible(lbpugt_4,732). happensAt(inactive(lbpugt_4),732).
orientation(lbpugt_2,101,733). coords(lbpugt_2,293,97,733).  visible(lbpugt_2,733). happensAt(walking(lbpugt_2),733). orientation(lbpugt_4,135,733). coords(lbpugt_4,219,34,733).  visible(lbpugt_4,733). happensAt(inactive(lbpugt_4),733).
orientation(lbpugt_2,97,734). coords(lbpugt_2,294,97,734).  visible(lbpugt_2,734). happensAt(walking(lbpugt_2),734). orientation(lbpugt_4,135,734). coords(lbpugt_4,219,34,734).  visible(lbpugt_4,734). happensAt(inactive(lbpugt_4),734).
orientation(lbpugt_2,97,735). coords(lbpugt_2,294,97,735).  visible(lbpugt_2,735). happensAt(walking(lbpugt_2),735). orientation(lbpugt_4,135,735). coords(lbpugt_4,219,34,735).  visible(lbpugt_4,735). happensAt(inactive(lbpugt_4),735).
orientation(lbpugt_2,97,736). coords(lbpugt_2,294,97,736).  visible(lbpugt_2,736). happensAt(walking(lbpugt_2),736). orientation(lbpugt_4,135,736). coords(lbpugt_4,219,34,736).  visible(lbpugt_4,736). happensAt(inactive(lbpugt_4),736).
orientation(lbpugt_2,97,737). coords(lbpugt_2,294,97,737).  visible(lbpugt_2,737). happensAt(walking(lbpugt_2),737). orientation(lbpugt_4,135,737). coords(lbpugt_4,219,34,737).  visible(lbpugt_4,737). happensAt(inactive(lbpugt_4),737).
orientation(lbpugt_2,97,738). coords(lbpugt_2,294,98,738).  visible(lbpugt_2,738). happensAt(active(lbpugt_2),738). orientation(lbpugt_4,135,738). coords(lbpugt_4,219,34,738).  visible(lbpugt_4,738). happensAt(inactive(lbpugt_4),738).
orientation(lbpugt_2,97,739). coords(lbpugt_2,294,98,739).  visible(lbpugt_2,739). happensAt(active(lbpugt_2),739). orientation(lbpugt_4,135,739). coords(lbpugt_4,219,34,739).  visible(lbpugt_4,739). happensAt(inactive(lbpugt_4),739).
orientation(lbpugt_2,97,740). coords(lbpugt_2,294,98,740).  visible(lbpugt_2,740). happensAt(active(lbpugt_2),740). orientation(lbpugt_4,135,740). coords(lbpugt_4,219,34,740).  visible(lbpugt_4,740). happensAt(inactive(lbpugt_4),740).
orientation(lbpugt_2,97,741). coords(lbpugt_2,294,98,741).  visible(lbpugt_2,741). happensAt(active(lbpugt_2),741). orientation(lbpugt_4,135,741). coords(lbpugt_4,219,34,741).  visible(lbpugt_4,741). happensAt(inactive(lbpugt_4),741).
orientation(lbpugt_2,97,742). coords(lbpugt_2,294,98,742).  visible(lbpugt_2,742). happensAt(active(lbpugt_2),742). orientation(lbpugt_4,135,742). coords(lbpugt_4,219,34,742).  visible(lbpugt_4,742). happensAt(inactive(lbpugt_4),742).
orientation(lbpugt_2,97,743). coords(lbpugt_2,294,98,743).  visible(lbpugt_2,743). happensAt(active(lbpugt_2),743). orientation(lbpugt_4,135,743). coords(lbpugt_4,219,34,743).  visible(lbpugt_4,743). happensAt(inactive(lbpugt_4),743).
#show holdsAt/2.