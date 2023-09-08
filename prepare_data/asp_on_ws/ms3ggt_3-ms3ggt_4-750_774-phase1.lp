
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

	orientation(ms3ggt_3,134,750). coords(ms3ggt_3,73,65,750).  happensAt(appear(ms3ggt_3),750). happensAt(walking(ms3ggt_3),750). orientation(ms3ggt_4,129,750). coords(ms3ggt_4,154,21,750).  visible(ms3ggt_4,750). happensAt(walking(ms3ggt_4),750).
orientation(ms3ggt_3,134,751). coords(ms3ggt_3,73,65,751).  visible(ms3ggt_3,751). happensAt(walking(ms3ggt_3),751). orientation(ms3ggt_4,129,751). coords(ms3ggt_4,153,21,751).  visible(ms3ggt_4,751). happensAt(walking(ms3ggt_4),751).
orientation(ms3ggt_3,134,752). coords(ms3ggt_3,73,67,752).  visible(ms3ggt_3,752). happensAt(walking(ms3ggt_3),752). orientation(ms3ggt_4,129,752). coords(ms3ggt_4,153,21,752).  visible(ms3ggt_4,752). happensAt(walking(ms3ggt_4),752).
orientation(ms3ggt_3,134,753). coords(ms3ggt_3,73,68,753).  visible(ms3ggt_3,753). happensAt(walking(ms3ggt_3),753). orientation(ms3ggt_4,129,753). coords(ms3ggt_4,153,21,753).  visible(ms3ggt_4,753). happensAt(walking(ms3ggt_4),753).
orientation(ms3ggt_3,142,754). coords(ms3ggt_3,72,69,754).  visible(ms3ggt_3,754). happensAt(walking(ms3ggt_3),754). orientation(ms3ggt_4,129,754). coords(ms3ggt_4,153,21,754).  visible(ms3ggt_4,754). happensAt(walking(ms3ggt_4),754).
orientation(ms3ggt_3,142,755). coords(ms3ggt_3,72,70,755).  visible(ms3ggt_3,755). happensAt(walking(ms3ggt_3),755). orientation(ms3ggt_4,129,755). coords(ms3ggt_4,153,21,755).  visible(ms3ggt_4,755). happensAt(walking(ms3ggt_4),755).
orientation(ms3ggt_3,142,756). coords(ms3ggt_3,71,70,756).  visible(ms3ggt_3,756). happensAt(walking(ms3ggt_3),756). orientation(ms3ggt_4,129,756). coords(ms3ggt_4,152,21,756).  visible(ms3ggt_4,756). happensAt(walking(ms3ggt_4),756).
orientation(ms3ggt_3,142,757). coords(ms3ggt_3,71,71,757).  visible(ms3ggt_3,757). happensAt(walking(ms3ggt_3),757). orientation(ms3ggt_4,129,757). coords(ms3ggt_4,151,21,757).  visible(ms3ggt_4,757). happensAt(walking(ms3ggt_4),757).
orientation(ms3ggt_3,142,758). coords(ms3ggt_3,71,73,758).  visible(ms3ggt_3,758). happensAt(walking(ms3ggt_3),758). orientation(ms3ggt_4,129,758). coords(ms3ggt_4,150,22,758).  visible(ms3ggt_4,758). happensAt(walking(ms3ggt_4),758).
orientation(ms3ggt_3,142,759). coords(ms3ggt_3,71,74,759).  visible(ms3ggt_3,759). happensAt(walking(ms3ggt_3),759). orientation(ms3ggt_4,129,759). coords(ms3ggt_4,150,22,759).  visible(ms3ggt_4,759). happensAt(walking(ms3ggt_4),759).
orientation(ms3ggt_3,142,760). coords(ms3ggt_3,71,75,760).  visible(ms3ggt_3,760). happensAt(walking(ms3ggt_3),760). orientation(ms3ggt_4,129,760). coords(ms3ggt_4,150,22,760).  visible(ms3ggt_4,760). happensAt(walking(ms3ggt_4),760).
orientation(ms3ggt_3,142,761). coords(ms3ggt_3,71,76,761).  visible(ms3ggt_3,761). happensAt(walking(ms3ggt_3),761). orientation(ms3ggt_4,129,761). coords(ms3ggt_4,150,22,761).  visible(ms3ggt_4,761). happensAt(walking(ms3ggt_4),761).
orientation(ms3ggt_3,148,762). coords(ms3ggt_3,71,77,762).  visible(ms3ggt_3,762). happensAt(walking(ms3ggt_3),762). orientation(ms3ggt_4,129,762). coords(ms3ggt_4,150,22,762).  visible(ms3ggt_4,762). happensAt(walking(ms3ggt_4),762).
orientation(ms3ggt_3,148,763). coords(ms3ggt_3,70,77,763).  visible(ms3ggt_3,763). happensAt(walking(ms3ggt_3),763). orientation(ms3ggt_4,129,763). coords(ms3ggt_4,150,22,763).  visible(ms3ggt_4,763). happensAt(walking(ms3ggt_4),763).
orientation(ms3ggt_3,148,764). coords(ms3ggt_3,68,78,764).  visible(ms3ggt_3,764). happensAt(walking(ms3ggt_3),764). orientation(ms3ggt_4,129,764). coords(ms3ggt_4,150,22,764).  visible(ms3ggt_4,764). happensAt(walking(ms3ggt_4),764).
orientation(ms3ggt_3,148,765). coords(ms3ggt_3,66,79,765).  visible(ms3ggt_3,765). happensAt(walking(ms3ggt_3),765). orientation(ms3ggt_4,129,765). coords(ms3ggt_4,149,23,765).  visible(ms3ggt_4,765). happensAt(walking(ms3ggt_4),765).
orientation(ms3ggt_3,140,766). coords(ms3ggt_3,65,81,766).  visible(ms3ggt_3,766). happensAt(walking(ms3ggt_3),766). orientation(ms3ggt_4,129,766). coords(ms3ggt_4,149,23,766).  visible(ms3ggt_4,766). happensAt(walking(ms3ggt_4),766).
orientation(ms3ggt_3,140,767). coords(ms3ggt_3,65,81,767).  visible(ms3ggt_3,767). happensAt(walking(ms3ggt_3),767). orientation(ms3ggt_4,129,767). coords(ms3ggt_4,149,24,767).  visible(ms3ggt_4,767). happensAt(walking(ms3ggt_4),767).
orientation(ms3ggt_3,140,768). coords(ms3ggt_3,65,82,768).  visible(ms3ggt_3,768). happensAt(walking(ms3ggt_3),768). orientation(ms3ggt_4,129,768). coords(ms3ggt_4,148,24,768).  visible(ms3ggt_4,768). happensAt(walking(ms3ggt_4),768).
orientation(ms3ggt_3,148,769). coords(ms3ggt_3,64,82,769).  visible(ms3ggt_3,769). happensAt(walking(ms3ggt_3),769). orientation(ms3ggt_4,120,769). coords(ms3ggt_4,148,24,769).  visible(ms3ggt_4,769). happensAt(walking(ms3ggt_4),769).
orientation(ms3ggt_3,148,770). coords(ms3ggt_3,64,82,770).  visible(ms3ggt_3,770). happensAt(walking(ms3ggt_3),770). orientation(ms3ggt_4,120,770). coords(ms3ggt_4,147,24,770).  visible(ms3ggt_4,770). happensAt(walking(ms3ggt_4),770).
orientation(ms3ggt_3,148,771). coords(ms3ggt_3,64,83,771).  visible(ms3ggt_3,771). happensAt(walking(ms3ggt_3),771). orientation(ms3ggt_4,120,771). coords(ms3ggt_4,147,24,771).  visible(ms3ggt_4,771). happensAt(walking(ms3ggt_4),771).
orientation(ms3ggt_3,148,772). coords(ms3ggt_3,65,84,772).  visible(ms3ggt_3,772). happensAt(walking(ms3ggt_3),772). orientation(ms3ggt_4,120,772). coords(ms3ggt_4,147,24,772).  visible(ms3ggt_4,772). happensAt(walking(ms3ggt_4),772).
orientation(ms3ggt_3,148,773). coords(ms3ggt_3,63,86,773).  visible(ms3ggt_3,773). happensAt(walking(ms3ggt_3),773). orientation(ms3ggt_4,120,773). coords(ms3ggt_4,147,24,773).  visible(ms3ggt_4,773). happensAt(walking(ms3ggt_4),773).
orientation(ms3ggt_3,148,774). coords(ms3ggt_3,60,87,774).  visible(ms3ggt_3,774). happensAt(walking(ms3ggt_3),774). orientation(ms3ggt_4,120,774). coords(ms3ggt_4,146,24,774).  visible(ms3ggt_4,774). happensAt(walking(ms3ggt_4),774).
#show holdsAt/2.