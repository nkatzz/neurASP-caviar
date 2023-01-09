
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
orientation(lbgt_3,61,752). coords(lbgt_3,146,162,752).  visible(lbgt_3,752). happensAt(inactive(lbgt_3),752). orientation(lbgt_4,153,752). coords(lbgt_4,90,133,752).  visible(lbgt_4,752). happensAt(walking(lbgt_4),752).
orientation(lbgt_3,61,753). coords(lbgt_3,146,162,753).  visible(lbgt_3,753). happensAt(inactive(lbgt_3),753). orientation(lbgt_4,153,753). coords(lbgt_4,90,133,753).  visible(lbgt_4,753). happensAt(walking(lbgt_4),753).
orientation(lbgt_3,61,754). coords(lbgt_3,146,162,754).  visible(lbgt_3,754). happensAt(inactive(lbgt_3),754). orientation(lbgt_4,153,754). coords(lbgt_4,90,133,754).  visible(lbgt_4,754). happensAt(walking(lbgt_4),754).
orientation(lbgt_3,61,755). coords(lbgt_3,146,162,755).  visible(lbgt_3,755). happensAt(inactive(lbgt_3),755). orientation(lbgt_4,153,755). coords(lbgt_4,89,132,755).  visible(lbgt_4,755). happensAt(walking(lbgt_4),755).
orientation(lbgt_3,61,756). coords(lbgt_3,146,162,756).  visible(lbgt_3,756). happensAt(inactive(lbgt_3),756). orientation(lbgt_4,153,756). coords(lbgt_4,89,132,756).  visible(lbgt_4,756). happensAt(walking(lbgt_4),756).
orientation(lbgt_3,61,757). coords(lbgt_3,146,162,757).  visible(lbgt_3,757). happensAt(inactive(lbgt_3),757). orientation(lbgt_4,153,757). coords(lbgt_4,89,131,757).  visible(lbgt_4,757). happensAt(walking(lbgt_4),757).
orientation(lbgt_3,61,758). coords(lbgt_3,146,162,758).  visible(lbgt_3,758). happensAt(inactive(lbgt_3),758). orientation(lbgt_4,153,758). coords(lbgt_4,89,130,758).  visible(lbgt_4,758). happensAt(walking(lbgt_4),758).
orientation(lbgt_3,61,759). coords(lbgt_3,146,162,759).  visible(lbgt_3,759). happensAt(inactive(lbgt_3),759). orientation(lbgt_4,153,759). coords(lbgt_4,89,129,759).  visible(lbgt_4,759). happensAt(walking(lbgt_4),759).
orientation(lbgt_3,61,760). coords(lbgt_3,146,162,760).  visible(lbgt_3,760). happensAt(inactive(lbgt_3),760). orientation(lbgt_4,153,760). coords(lbgt_4,87,128,760).  visible(lbgt_4,760). happensAt(walking(lbgt_4),760).
orientation(lbgt_3,61,761). coords(lbgt_3,146,162,761).  visible(lbgt_3,761). happensAt(inactive(lbgt_3),761). orientation(lbgt_4,153,761). coords(lbgt_4,87,126,761).  visible(lbgt_4,761). happensAt(walking(lbgt_4),761).
orientation(lbgt_3,61,762). coords(lbgt_3,146,162,762).  visible(lbgt_3,762). happensAt(inactive(lbgt_3),762). orientation(lbgt_4,153,762). coords(lbgt_4,87,125,762).  visible(lbgt_4,762). happensAt(walking(lbgt_4),762).
orientation(lbgt_3,61,763). coords(lbgt_3,146,162,763).  visible(lbgt_3,763). happensAt(inactive(lbgt_3),763). orientation(lbgt_4,153,763). coords(lbgt_4,87,125,763).  visible(lbgt_4,763). happensAt(walking(lbgt_4),763).
#show holdsAt/2.