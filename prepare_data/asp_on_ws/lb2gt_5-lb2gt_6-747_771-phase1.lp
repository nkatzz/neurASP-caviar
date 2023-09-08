
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

	orientation(lb2gt_5,132,747). coords(lb2gt_5,168,21,747).  happensAt(appear(lb2gt_5),747). happensAt(walking(lb2gt_5),747). orientation(lb2gt_6,151,747). coords(lb2gt_6,72,122,747).  visible(lb2gt_6,747). happensAt(inactive(lb2gt_6),747).
orientation(lb2gt_5,132,748). coords(lb2gt_5,168,21,748).  visible(lb2gt_5,748). happensAt(walking(lb2gt_5),748). orientation(lb2gt_6,151,748). coords(lb2gt_6,72,122,748).  visible(lb2gt_6,748). happensAt(inactive(lb2gt_6),748).
orientation(lb2gt_5,132,749). coords(lb2gt_5,168,24,749).  visible(lb2gt_5,749). happensAt(walking(lb2gt_5),749). orientation(lb2gt_6,151,749). coords(lb2gt_6,72,122,749).  visible(lb2gt_6,749). happensAt(inactive(lb2gt_6),749).
orientation(lb2gt_5,132,750). coords(lb2gt_5,167,26,750).  visible(lb2gt_5,750). happensAt(walking(lb2gt_5),750). orientation(lb2gt_6,151,750). coords(lb2gt_6,72,122,750).  visible(lb2gt_6,750). happensAt(inactive(lb2gt_6),750).
orientation(lb2gt_5,132,751). coords(lb2gt_5,167,26,751).  visible(lb2gt_5,751). happensAt(walking(lb2gt_5),751). orientation(lb2gt_6,151,751). coords(lb2gt_6,72,122,751).  visible(lb2gt_6,751). happensAt(inactive(lb2gt_6),751).
orientation(lb2gt_5,132,752). coords(lb2gt_5,167,27,752).  visible(lb2gt_5,752). happensAt(walking(lb2gt_5),752). orientation(lb2gt_6,151,752). coords(lb2gt_6,72,122,752).  visible(lb2gt_6,752). happensAt(inactive(lb2gt_6),752).
orientation(lb2gt_5,132,753). coords(lb2gt_5,167,28,753).  visible(lb2gt_5,753). happensAt(walking(lb2gt_5),753). orientation(lb2gt_6,151,753). coords(lb2gt_6,72,122,753).  visible(lb2gt_6,753). happensAt(inactive(lb2gt_6),753).
orientation(lb2gt_5,132,754). coords(lb2gt_5,166,29,754).  visible(lb2gt_5,754). happensAt(walking(lb2gt_5),754). orientation(lb2gt_6,151,754). coords(lb2gt_6,72,122,754).  visible(lb2gt_6,754). happensAt(inactive(lb2gt_6),754).
orientation(lb2gt_5,132,755). coords(lb2gt_5,166,29,755).  visible(lb2gt_5,755). happensAt(walking(lb2gt_5),755). orientation(lb2gt_6,151,755). coords(lb2gt_6,72,122,755).  visible(lb2gt_6,755). happensAt(inactive(lb2gt_6),755).
orientation(lb2gt_5,132,756). coords(lb2gt_5,165,29,756).  visible(lb2gt_5,756). happensAt(walking(lb2gt_5),756). orientation(lb2gt_6,151,756). coords(lb2gt_6,72,122,756).  visible(lb2gt_6,756). happensAt(inactive(lb2gt_6),756).
orientation(lb2gt_5,132,757). coords(lb2gt_5,165,29,757).  visible(lb2gt_5,757). happensAt(walking(lb2gt_5),757). orientation(lb2gt_6,151,757). coords(lb2gt_6,72,122,757).  visible(lb2gt_6,757). happensAt(inactive(lb2gt_6),757).
orientation(lb2gt_5,132,758). coords(lb2gt_5,164,29,758).  visible(lb2gt_5,758). happensAt(walking(lb2gt_5),758). orientation(lb2gt_6,151,758). coords(lb2gt_6,72,122,758).  visible(lb2gt_6,758). happensAt(inactive(lb2gt_6),758).
orientation(lb2gt_5,132,759). coords(lb2gt_5,164,29,759).  visible(lb2gt_5,759). happensAt(walking(lb2gt_5),759). orientation(lb2gt_6,151,759). coords(lb2gt_6,72,122,759).  visible(lb2gt_6,759). happensAt(inactive(lb2gt_6),759).
orientation(lb2gt_5,132,760). coords(lb2gt_5,164,29,760).  visible(lb2gt_5,760). happensAt(walking(lb2gt_5),760). orientation(lb2gt_6,151,760). coords(lb2gt_6,72,122,760).  visible(lb2gt_6,760). happensAt(inactive(lb2gt_6),760).
orientation(lb2gt_5,132,761). coords(lb2gt_5,163,30,761).  visible(lb2gt_5,761). happensAt(walking(lb2gt_5),761). orientation(lb2gt_6,151,761). coords(lb2gt_6,72,122,761).  visible(lb2gt_6,761). happensAt(inactive(lb2gt_6),761).
orientation(lb2gt_5,132,762). coords(lb2gt_5,163,30,762).  visible(lb2gt_5,762). happensAt(walking(lb2gt_5),762). orientation(lb2gt_6,151,762). coords(lb2gt_6,72,122,762).  visible(lb2gt_6,762). happensAt(inactive(lb2gt_6),762).
orientation(lb2gt_5,132,763). coords(lb2gt_5,162,30,763).  visible(lb2gt_5,763). happensAt(walking(lb2gt_5),763). orientation(lb2gt_6,151,763). coords(lb2gt_6,72,122,763).  visible(lb2gt_6,763). happensAt(inactive(lb2gt_6),763).
orientation(lb2gt_5,132,764). coords(lb2gt_5,162,30,764).  visible(lb2gt_5,764). happensAt(walking(lb2gt_5),764). orientation(lb2gt_6,151,764). coords(lb2gt_6,72,122,764).  visible(lb2gt_6,764). happensAt(inactive(lb2gt_6),764).
orientation(lb2gt_5,132,765). coords(lb2gt_5,162,32,765).  visible(lb2gt_5,765). happensAt(walking(lb2gt_5),765). orientation(lb2gt_6,151,765). coords(lb2gt_6,72,122,765).  visible(lb2gt_6,765). happensAt(inactive(lb2gt_6),765).
orientation(lb2gt_5,132,766). coords(lb2gt_5,162,33,766).  visible(lb2gt_5,766). happensAt(walking(lb2gt_5),766). orientation(lb2gt_6,151,766). coords(lb2gt_6,72,122,766).  visible(lb2gt_6,766). happensAt(inactive(lb2gt_6),766).
orientation(lb2gt_5,132,767). coords(lb2gt_5,161,34,767).  visible(lb2gt_5,767). happensAt(walking(lb2gt_5),767). orientation(lb2gt_6,151,767). coords(lb2gt_6,72,122,767).  visible(lb2gt_6,767). happensAt(inactive(lb2gt_6),767).
orientation(lb2gt_5,132,768). coords(lb2gt_5,161,35,768).  visible(lb2gt_5,768). happensAt(walking(lb2gt_5),768). orientation(lb2gt_6,151,768). coords(lb2gt_6,72,122,768).  visible(lb2gt_6,768). happensAt(inactive(lb2gt_6),768).
orientation(lb2gt_5,132,769). coords(lb2gt_5,159,34,769).  visible(lb2gt_5,769). happensAt(walking(lb2gt_5),769). orientation(lb2gt_6,151,769). coords(lb2gt_6,72,122,769).  visible(lb2gt_6,769). happensAt(inactive(lb2gt_6),769).
orientation(lb2gt_5,132,770). coords(lb2gt_5,159,34,770).  visible(lb2gt_5,770). happensAt(walking(lb2gt_5),770). orientation(lb2gt_6,151,770). coords(lb2gt_6,72,122,770).  visible(lb2gt_6,770). happensAt(inactive(lb2gt_6),770).
orientation(lb2gt_5,132,771). coords(lb2gt_5,159,34,771).  visible(lb2gt_5,771). happensAt(walking(lb2gt_5),771). orientation(lb2gt_6,151,771). coords(lb2gt_6,72,122,771).  visible(lb2gt_6,771). happensAt(inactive(lb2gt_6),771).
#show holdsAt/2.