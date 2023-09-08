
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
orientation(br1gt_2,12,755). coords(br1gt_2,86,209,755).  visible(br1gt_2,755). happensAt(inactive(br1gt_2),755). orientation(br1gt_3,89,755). coords(br1gt_3,303,106,755).  visible(br1gt_3,755). happensAt(active(br1gt_3),755).
orientation(br1gt_2,12,756). coords(br1gt_2,86,209,756).  visible(br1gt_2,756). happensAt(inactive(br1gt_2),756). orientation(br1gt_3,89,756). coords(br1gt_3,303,107,756).  visible(br1gt_3,756). happensAt(walking(br1gt_3),756).
orientation(br1gt_2,12,757). coords(br1gt_2,86,209,757).  visible(br1gt_2,757). happensAt(inactive(br1gt_2),757). orientation(br1gt_3,89,757). coords(br1gt_3,302,107,757).  visible(br1gt_3,757). happensAt(walking(br1gt_3),757).
orientation(br1gt_2,12,758). coords(br1gt_2,86,209,758).  visible(br1gt_2,758). happensAt(inactive(br1gt_2),758). orientation(br1gt_3,89,758). coords(br1gt_3,302,107,758).  visible(br1gt_3,758). happensAt(walking(br1gt_3),758).
orientation(br1gt_2,12,759). coords(br1gt_2,86,209,759).  visible(br1gt_2,759). happensAt(inactive(br1gt_2),759). orientation(br1gt_3,89,759). coords(br1gt_3,302,108,759).  visible(br1gt_3,759). happensAt(walking(br1gt_3),759).
orientation(br1gt_2,12,760). coords(br1gt_2,86,209,760).  visible(br1gt_2,760). happensAt(inactive(br1gt_2),760). orientation(br1gt_3,89,760). coords(br1gt_3,301,108,760).  visible(br1gt_3,760). happensAt(walking(br1gt_3),760).
orientation(br1gt_2,12,761). coords(br1gt_2,86,209,761).  visible(br1gt_2,761). happensAt(inactive(br1gt_2),761). orientation(br1gt_3,89,761). coords(br1gt_3,301,109,761).  visible(br1gt_3,761). happensAt(walking(br1gt_3),761).
orientation(br1gt_2,12,762). coords(br1gt_2,86,209,762).  visible(br1gt_2,762). happensAt(inactive(br1gt_2),762). orientation(br1gt_3,89,762). coords(br1gt_3,301,109,762).  visible(br1gt_3,762). happensAt(walking(br1gt_3),762).
orientation(br1gt_2,12,763). coords(br1gt_2,86,209,763).  visible(br1gt_2,763). happensAt(inactive(br1gt_2),763). orientation(br1gt_3,89,763). coords(br1gt_3,301,110,763).  visible(br1gt_3,763). happensAt(walking(br1gt_3),763).
orientation(br1gt_2,12,764). coords(br1gt_2,86,209,764).  visible(br1gt_2,764). happensAt(inactive(br1gt_2),764). orientation(br1gt_3,89,764). coords(br1gt_3,301,113,764).  visible(br1gt_3,764). happensAt(walking(br1gt_3),764).
orientation(br1gt_2,12,765). coords(br1gt_2,86,209,765).  visible(br1gt_2,765). happensAt(inactive(br1gt_2),765). orientation(br1gt_3,107,765). coords(br1gt_3,301,115,765).  visible(br1gt_3,765). happensAt(walking(br1gt_3),765).
orientation(br1gt_2,12,766). coords(br1gt_2,86,209,766).  visible(br1gt_2,766). happensAt(inactive(br1gt_2),766). orientation(br1gt_3,107,766). coords(br1gt_3,301,115,766).  visible(br1gt_3,766). happensAt(walking(br1gt_3),766).
#show holdsAt/2.