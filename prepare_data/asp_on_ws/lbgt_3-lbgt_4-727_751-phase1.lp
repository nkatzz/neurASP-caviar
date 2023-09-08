
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

	orientation(lbgt_3,61,727). coords(lbgt_3,146,162,727).  visible(lbgt_3,727). happensAt(inactive(lbgt_3),727). orientation(lbgt_4,158,727). coords(lbgt_4,101,154,727).  visible(lbgt_4,727). happensAt(walking(lbgt_4),727).
orientation(lbgt_3,61,728). coords(lbgt_3,146,162,728).  visible(lbgt_3,728). happensAt(inactive(lbgt_3),728). orientation(lbgt_4,158,728). coords(lbgt_4,101,152,728).  visible(lbgt_4,728). happensAt(walking(lbgt_4),728).
orientation(lbgt_3,61,729). coords(lbgt_3,146,162,729).  visible(lbgt_3,729). happensAt(inactive(lbgt_3),729). orientation(lbgt_4,158,729). coords(lbgt_4,100,152,729).  visible(lbgt_4,729). happensAt(walking(lbgt_4),729).
orientation(lbgt_3,61,730). coords(lbgt_3,146,162,730).  visible(lbgt_3,730). happensAt(inactive(lbgt_3),730). orientation(lbgt_4,158,730). coords(lbgt_4,100,151,730).  visible(lbgt_4,730). happensAt(walking(lbgt_4),730).
orientation(lbgt_3,61,731). coords(lbgt_3,146,162,731).  visible(lbgt_3,731). happensAt(inactive(lbgt_3),731). orientation(lbgt_4,158,731). coords(lbgt_4,100,151,731).  visible(lbgt_4,731). happensAt(walking(lbgt_4),731).
orientation(lbgt_3,61,732). coords(lbgt_3,146,162,732).  visible(lbgt_3,732). happensAt(inactive(lbgt_3),732). orientation(lbgt_4,153,732). coords(lbgt_4,100,151,732).  visible(lbgt_4,732). happensAt(walking(lbgt_4),732).
orientation(lbgt_3,61,733). coords(lbgt_3,146,162,733).  visible(lbgt_3,733). happensAt(inactive(lbgt_3),733). orientation(lbgt_4,158,733). coords(lbgt_4,100,151,733).  visible(lbgt_4,733). happensAt(walking(lbgt_4),733).
orientation(lbgt_3,61,734). coords(lbgt_3,146,162,734).  visible(lbgt_3,734). happensAt(inactive(lbgt_3),734). orientation(lbgt_4,159,734). coords(lbgt_4,100,150,734).  visible(lbgt_4,734). happensAt(walking(lbgt_4),734).
orientation(lbgt_3,61,735). coords(lbgt_3,146,162,735).  visible(lbgt_3,735). happensAt(inactive(lbgt_3),735). orientation(lbgt_4,159,735). coords(lbgt_4,99,148,735).  visible(lbgt_4,735). happensAt(walking(lbgt_4),735).
orientation(lbgt_3,61,736). coords(lbgt_3,146,162,736).  visible(lbgt_3,736). happensAt(inactive(lbgt_3),736). orientation(lbgt_4,160,736). coords(lbgt_4,98,147,736).  visible(lbgt_4,736). happensAt(walking(lbgt_4),736).
orientation(lbgt_3,61,737). coords(lbgt_3,146,162,737).  visible(lbgt_3,737). happensAt(inactive(lbgt_3),737). orientation(lbgt_4,159,737). coords(lbgt_4,96,146,737).  visible(lbgt_4,737). happensAt(walking(lbgt_4),737).
orientation(lbgt_3,61,738). coords(lbgt_3,146,162,738).  visible(lbgt_3,738). happensAt(inactive(lbgt_3),738). orientation(lbgt_4,159,738). coords(lbgt_4,95,144,738).  visible(lbgt_4,738). happensAt(walking(lbgt_4),738).
orientation(lbgt_3,61,739). coords(lbgt_3,146,162,739).  visible(lbgt_3,739). happensAt(inactive(lbgt_3),739). orientation(lbgt_4,159,739). coords(lbgt_4,95,144,739).  visible(lbgt_4,739). happensAt(walking(lbgt_4),739).
orientation(lbgt_3,61,740). coords(lbgt_3,146,162,740).  visible(lbgt_3,740). happensAt(inactive(lbgt_3),740). orientation(lbgt_4,157,740). coords(lbgt_4,95,143,740).  visible(lbgt_4,740). happensAt(walking(lbgt_4),740).
orientation(lbgt_3,61,741). coords(lbgt_3,146,162,741).  visible(lbgt_3,741). happensAt(inactive(lbgt_3),741). orientation(lbgt_4,153,741). coords(lbgt_4,95,143,741).  visible(lbgt_4,741). happensAt(walking(lbgt_4),741).
orientation(lbgt_3,61,742). coords(lbgt_3,146,162,742).  visible(lbgt_3,742). happensAt(inactive(lbgt_3),742). orientation(lbgt_4,152,742). coords(lbgt_4,94,143,742).  visible(lbgt_4,742). happensAt(walking(lbgt_4),742).
orientation(lbgt_3,61,743). coords(lbgt_3,146,162,743).  visible(lbgt_3,743). happensAt(inactive(lbgt_3),743). orientation(lbgt_4,152,743). coords(lbgt_4,94,142,743).  visible(lbgt_4,743). happensAt(walking(lbgt_4),743).
orientation(lbgt_3,61,744). coords(lbgt_3,146,162,744).  visible(lbgt_3,744). happensAt(inactive(lbgt_3),744). orientation(lbgt_4,150,744). coords(lbgt_4,94,142,744).  visible(lbgt_4,744). happensAt(walking(lbgt_4),744).
orientation(lbgt_3,61,745). coords(lbgt_3,146,162,745).  visible(lbgt_3,745). happensAt(inactive(lbgt_3),745). orientation(lbgt_4,149,745). coords(lbgt_4,94,142,745).  visible(lbgt_4,745). happensAt(walking(lbgt_4),745).
orientation(lbgt_3,61,746). coords(lbgt_3,146,162,746).  visible(lbgt_3,746). happensAt(inactive(lbgt_3),746). orientation(lbgt_4,147,746). coords(lbgt_4,93,140,746).  visible(lbgt_4,746). happensAt(walking(lbgt_4),746).
orientation(lbgt_3,61,747). coords(lbgt_3,146,162,747).  visible(lbgt_3,747). happensAt(inactive(lbgt_3),747). orientation(lbgt_4,153,747). coords(lbgt_4,93,140,747).  visible(lbgt_4,747). happensAt(walking(lbgt_4),747).
orientation(lbgt_3,61,748). coords(lbgt_3,146,162,748).  visible(lbgt_3,748). happensAt(inactive(lbgt_3),748). orientation(lbgt_4,150,748). coords(lbgt_4,92,138,748).  visible(lbgt_4,748). happensAt(walking(lbgt_4),748).
orientation(lbgt_3,61,749). coords(lbgt_3,146,162,749).  visible(lbgt_3,749). happensAt(inactive(lbgt_3),749). orientation(lbgt_4,150,749). coords(lbgt_4,91,136,749).  visible(lbgt_4,749). happensAt(walking(lbgt_4),749).
orientation(lbgt_3,61,750). coords(lbgt_3,146,162,750).  visible(lbgt_3,750). happensAt(inactive(lbgt_3),750). orientation(lbgt_4,150,750). coords(lbgt_4,91,136,750).  visible(lbgt_4,750). happensAt(walking(lbgt_4),750).
orientation(lbgt_3,61,751). coords(lbgt_3,146,162,751).  visible(lbgt_3,751). happensAt(inactive(lbgt_3),751). orientation(lbgt_4,153,751). coords(lbgt_4,91,134,751).  visible(lbgt_4,751). happensAt(walking(lbgt_4),751).
#show holdsAt/2.