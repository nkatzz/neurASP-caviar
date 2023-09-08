
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
orientation(br1gt_2,12,767). coords(br1gt_2,86,209,767).  visible(br1gt_2,767). happensAt(inactive(br1gt_2),767). orientation(br1gt_3,107,767). coords(br1gt_3,300,121,767).  visible(br1gt_3,767). happensAt(walking(br1gt_3),767).
orientation(br1gt_2,12,768). coords(br1gt_2,86,209,768).  visible(br1gt_2,768). happensAt(inactive(br1gt_2),768). orientation(br1gt_3,107,768). coords(br1gt_3,300,123,768).  visible(br1gt_3,768). happensAt(walking(br1gt_3),768).
orientation(br1gt_2,12,769). coords(br1gt_2,86,209,769).  visible(br1gt_2,769). happensAt(inactive(br1gt_2),769). orientation(br1gt_3,107,769). coords(br1gt_3,300,125,769).  visible(br1gt_3,769). happensAt(walking(br1gt_3),769).
orientation(br1gt_2,12,770). coords(br1gt_2,86,209,770).  visible(br1gt_2,770). happensAt(inactive(br1gt_2),770). orientation(br1gt_3,107,770). coords(br1gt_3,300,127,770).  visible(br1gt_3,770). happensAt(walking(br1gt_3),770).
orientation(br1gt_2,12,771). coords(br1gt_2,86,209,771).  visible(br1gt_2,771). happensAt(inactive(br1gt_2),771). orientation(br1gt_3,107,771). coords(br1gt_3,300,128,771).  visible(br1gt_3,771). happensAt(walking(br1gt_3),771).
orientation(br1gt_2,12,772). coords(br1gt_2,86,209,772).  visible(br1gt_2,772). happensAt(inactive(br1gt_2),772). orientation(br1gt_3,107,772). coords(br1gt_3,300,130,772).  visible(br1gt_3,772). happensAt(walking(br1gt_3),772).
orientation(br1gt_2,12,773). coords(br1gt_2,86,209,773).  visible(br1gt_2,773). happensAt(inactive(br1gt_2),773). orientation(br1gt_3,107,773). coords(br1gt_3,300,131,773).  visible(br1gt_3,773). happensAt(walking(br1gt_3),773).
orientation(br1gt_2,12,774). coords(br1gt_2,86,209,774).  visible(br1gt_2,774). happensAt(inactive(br1gt_2),774). orientation(br1gt_3,107,774). coords(br1gt_3,301,132,774).  visible(br1gt_3,774). happensAt(walking(br1gt_3),774).
orientation(br1gt_2,12,775). coords(br1gt_2,86,209,775).  visible(br1gt_2,775). happensAt(inactive(br1gt_2),775). orientation(br1gt_3,107,775). coords(br1gt_3,301,133,775).  visible(br1gt_3,775). happensAt(walking(br1gt_3),775).
orientation(br1gt_2,12,776). coords(br1gt_2,86,209,776).  visible(br1gt_2,776). happensAt(inactive(br1gt_2),776). orientation(br1gt_3,107,776). coords(br1gt_3,302,134,776).  visible(br1gt_3,776). happensAt(walking(br1gt_3),776).
orientation(br1gt_2,12,777). coords(br1gt_2,86,209,777).  visible(br1gt_2,777). happensAt(inactive(br1gt_2),777). orientation(br1gt_3,107,777). coords(br1gt_3,302,135,777).  visible(br1gt_3,777). happensAt(walking(br1gt_3),777).
orientation(br1gt_2,12,778). coords(br1gt_2,86,209,778).  visible(br1gt_2,778). happensAt(inactive(br1gt_2),778). orientation(br1gt_3,107,778). coords(br1gt_3,302,136,778).  visible(br1gt_3,778). happensAt(walking(br1gt_3),778).
#show holdsAt/2.