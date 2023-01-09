
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

	orientation(ms3ggt_0,25,724). coords(ms3ggt_0,236,249,724).  visible(ms3ggt_0,724). happensAt(walking(ms3ggt_0),724). orientation(ms3ggt_4,129,724). coords(ms3ggt_4,159,20,724).  visible(ms3ggt_4,724). happensAt(walking(ms3ggt_4),724).
orientation(ms3ggt_0,25,725). coords(ms3ggt_0,237,252,725).  visible(ms3ggt_0,725). happensAt(walking(ms3ggt_0),725). orientation(ms3ggt_4,129,725). coords(ms3ggt_4,159,20,725).  visible(ms3ggt_4,725). happensAt(walking(ms3ggt_4),725).
orientation(ms3ggt_0,26,726). coords(ms3ggt_0,240,254,726).  visible(ms3ggt_0,726). happensAt(walking(ms3ggt_0),726). orientation(ms3ggt_4,129,726). coords(ms3ggt_4,159,20,726).  visible(ms3ggt_4,726). happensAt(walking(ms3ggt_4),726).
orientation(ms3ggt_0,26,727). coords(ms3ggt_0,244,256,727).  visible(ms3ggt_0,727). happensAt(walking(ms3ggt_0),727). orientation(ms3ggt_4,129,727). coords(ms3ggt_4,158,20,727).  visible(ms3ggt_4,727). happensAt(walking(ms3ggt_4),727).
orientation(ms3ggt_0,21,728). coords(ms3ggt_0,249,256,728).  visible(ms3ggt_0,728). happensAt(walking(ms3ggt_0),728). orientation(ms3ggt_4,129,728). coords(ms3ggt_4,158,20,728).  visible(ms3ggt_4,728). happensAt(walking(ms3ggt_4),728).
orientation(ms3ggt_0,21,729). coords(ms3ggt_0,250,256,729).  visible(ms3ggt_0,729). happensAt(walking(ms3ggt_0),729). orientation(ms3ggt_4,129,729). coords(ms3ggt_4,158,20,729).  visible(ms3ggt_4,729). happensAt(walking(ms3ggt_4),729).
orientation(ms3ggt_0,21,730). coords(ms3ggt_0,253,257,730).  visible(ms3ggt_0,730). happensAt(walking(ms3ggt_0),730). orientation(ms3ggt_4,129,730). coords(ms3ggt_4,158,20,730).  visible(ms3ggt_4,730). happensAt(walking(ms3ggt_4),730).
orientation(ms3ggt_0,21,731). coords(ms3ggt_0,254,258,731).  visible(ms3ggt_0,731). happensAt(walking(ms3ggt_0),731). orientation(ms3ggt_4,129,731). coords(ms3ggt_4,158,20,731).  visible(ms3ggt_4,731). happensAt(walking(ms3ggt_4),731).
orientation(ms3ggt_0,28,732). coords(ms3ggt_0,255,259,732).  visible(ms3ggt_0,732). happensAt(walking(ms3ggt_0),732). orientation(ms3ggt_4,129,732). coords(ms3ggt_4,157,20,732).  visible(ms3ggt_4,732). happensAt(walking(ms3ggt_4),732).
orientation(ms3ggt_0,28,733). coords(ms3ggt_0,257,260,733).  visible(ms3ggt_0,733). happensAt(walking(ms3ggt_0),733). orientation(ms3ggt_4,129,733). coords(ms3ggt_4,157,20,733).  visible(ms3ggt_4,733). happensAt(walking(ms3ggt_4),733).
orientation(ms3ggt_0,28,734). coords(ms3ggt_0,257,262,734).  visible(ms3ggt_0,734). happensAt(walking(ms3ggt_0),734). orientation(ms3ggt_4,129,734). coords(ms3ggt_4,157,20,734).  visible(ms3ggt_4,734). happensAt(walking(ms3ggt_4),734).
orientation(ms3ggt_0,28,735). coords(ms3ggt_0,258,264,735).  visible(ms3ggt_0,735). happensAt(walking(ms3ggt_0),735). orientation(ms3ggt_4,129,735). coords(ms3ggt_4,157,20,735).  visible(ms3ggt_4,735). happensAt(walking(ms3ggt_4),735).
orientation(ms3ggt_0,28,736). coords(ms3ggt_0,259,266,736).  visible(ms3ggt_0,736). happensAt(walking(ms3ggt_0),736). orientation(ms3ggt_4,129,736). coords(ms3ggt_4,157,20,736).  visible(ms3ggt_4,736). happensAt(walking(ms3ggt_4),736).
orientation(ms3ggt_0,28,737). coords(ms3ggt_0,259,269,737).  visible(ms3ggt_0,737). happensAt(walking(ms3ggt_0),737). orientation(ms3ggt_4,129,737). coords(ms3ggt_4,157,21,737).  visible(ms3ggt_4,737). happensAt(walking(ms3ggt_4),737).
orientation(ms3ggt_0,28,738). coords(ms3ggt_0,259,271,738).  visible(ms3ggt_0,738). happensAt(walking(ms3ggt_0),738). orientation(ms3ggt_4,129,738). coords(ms3ggt_4,157,21,738).  visible(ms3ggt_4,738). happensAt(walking(ms3ggt_4),738).
orientation(ms3ggt_0,28,739). coords(ms3ggt_0,260,273,739).  visible(ms3ggt_0,739). happensAt(walking(ms3ggt_0),739). orientation(ms3ggt_4,129,739). coords(ms3ggt_4,157,21,739).  visible(ms3ggt_4,739). happensAt(walking(ms3ggt_4),739).
orientation(ms3ggt_0,36,740). coords(ms3ggt_0,260,273,740).  visible(ms3ggt_0,740). happensAt(walking(ms3ggt_0),740). orientation(ms3ggt_4,129,740). coords(ms3ggt_4,157,21,740).  visible(ms3ggt_4,740). happensAt(walking(ms3ggt_4),740).
orientation(ms3ggt_0,36,741). coords(ms3ggt_0,261,273,741).  visible(ms3ggt_0,741). happensAt(walking(ms3ggt_0),741). orientation(ms3ggt_4,129,741). coords(ms3ggt_4,156,21,741).  visible(ms3ggt_4,741). happensAt(walking(ms3ggt_4),741).
orientation(ms3ggt_0,36,742). coords(ms3ggt_0,261,273,742).  visible(ms3ggt_0,742). happensAt(walking(ms3ggt_0),742). orientation(ms3ggt_4,129,742). coords(ms3ggt_4,156,21,742).  visible(ms3ggt_4,742). happensAt(walking(ms3ggt_4),742).
orientation(ms3ggt_0,36,743). coords(ms3ggt_0,266,274,743).  visible(ms3ggt_0,743). happensAt(walking(ms3ggt_0),743). orientation(ms3ggt_4,129,743). coords(ms3ggt_4,156,21,743).  visible(ms3ggt_4,743). happensAt(walking(ms3ggt_4),743).
orientation(ms3ggt_0,41,744). coords(ms3ggt_0,267,273,744).  visible(ms3ggt_0,744). happensAt(walking(ms3ggt_0),744). orientation(ms3ggt_4,129,744). coords(ms3ggt_4,154,21,744).  visible(ms3ggt_4,744). happensAt(walking(ms3ggt_4),744).
orientation(ms3ggt_0,41,745). coords(ms3ggt_0,269,273,745).  visible(ms3ggt_0,745). happensAt(walking(ms3ggt_0),745). orientation(ms3ggt_4,129,745). coords(ms3ggt_4,154,21,745).  visible(ms3ggt_4,745). happensAt(walking(ms3ggt_4),745).
orientation(ms3ggt_0,78,746). coords(ms3ggt_0,271,273,746).  visible(ms3ggt_0,746). happensAt(walking(ms3ggt_0),746). orientation(ms3ggt_4,129,746). coords(ms3ggt_4,154,21,746).  visible(ms3ggt_4,746). happensAt(walking(ms3ggt_4),746).
orientation(ms3ggt_0,78,747). coords(ms3ggt_0,273,273,747).  visible(ms3ggt_0,747). happensAt(walking(ms3ggt_0),747). orientation(ms3ggt_4,129,747). coords(ms3ggt_4,154,21,747).  visible(ms3ggt_4,747). happensAt(walking(ms3ggt_4),747).
orientation(ms3ggt_0,78,748). coords(ms3ggt_0,274,275,748).  visible(ms3ggt_0,748). happensAt(walking(ms3ggt_0),748). orientation(ms3ggt_4,129,748). coords(ms3ggt_4,154,21,748).  visible(ms3ggt_4,748). happensAt(walking(ms3ggt_4),748).
#show holdsAt/2.