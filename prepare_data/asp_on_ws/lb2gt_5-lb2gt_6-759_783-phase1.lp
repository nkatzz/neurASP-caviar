
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
orientation(lb2gt_5,132,772). coords(lb2gt_5,158,35,772).  visible(lb2gt_5,772). happensAt(walking(lb2gt_5),772). orientation(lb2gt_6,151,772). coords(lb2gt_6,72,122,772).  visible(lb2gt_6,772). happensAt(inactive(lb2gt_6),772).
orientation(lb2gt_5,132,773). coords(lb2gt_5,157,35,773).  visible(lb2gt_5,773). happensAt(walking(lb2gt_5),773). orientation(lb2gt_6,151,773). coords(lb2gt_6,72,122,773).  visible(lb2gt_6,773). happensAt(inactive(lb2gt_6),773).
orientation(lb2gt_5,132,774). coords(lb2gt_5,155,35,774).  visible(lb2gt_5,774). happensAt(walking(lb2gt_5),774). orientation(lb2gt_6,151,774). coords(lb2gt_6,72,122,774).  visible(lb2gt_6,774). happensAt(inactive(lb2gt_6),774).
orientation(lb2gt_5,132,775). coords(lb2gt_5,154,35,775).  visible(lb2gt_5,775). happensAt(walking(lb2gt_5),775). orientation(lb2gt_6,151,775). coords(lb2gt_6,72,122,775).  visible(lb2gt_6,775). happensAt(inactive(lb2gt_6),775).
orientation(lb2gt_5,132,776). coords(lb2gt_5,153,35,776).  visible(lb2gt_5,776). happensAt(walking(lb2gt_5),776). orientation(lb2gt_6,151,776). coords(lb2gt_6,72,122,776).  visible(lb2gt_6,776). happensAt(inactive(lb2gt_6),776).
orientation(lb2gt_5,132,777). coords(lb2gt_5,152,35,777).  visible(lb2gt_5,777). happensAt(walking(lb2gt_5),777). orientation(lb2gt_6,151,777). coords(lb2gt_6,72,122,777).  visible(lb2gt_6,777). happensAt(inactive(lb2gt_6),777).
orientation(lb2gt_5,132,778). coords(lb2gt_5,152,36,778).  visible(lb2gt_5,778). happensAt(walking(lb2gt_5),778). orientation(lb2gt_6,151,778). coords(lb2gt_6,72,122,778).  visible(lb2gt_6,778). happensAt(inactive(lb2gt_6),778).
orientation(lb2gt_5,132,779). coords(lb2gt_5,152,36,779).  visible(lb2gt_5,779). happensAt(walking(lb2gt_5),779). orientation(lb2gt_6,151,779). coords(lb2gt_6,72,122,779).  visible(lb2gt_6,779). happensAt(inactive(lb2gt_6),779).
orientation(lb2gt_5,132,780). coords(lb2gt_5,152,38,780).  visible(lb2gt_5,780). happensAt(walking(lb2gt_5),780). orientation(lb2gt_6,151,780). coords(lb2gt_6,72,122,780).  visible(lb2gt_6,780). happensAt(inactive(lb2gt_6),780).
orientation(lb2gt_5,132,781). coords(lb2gt_5,152,39,781).  visible(lb2gt_5,781). happensAt(walking(lb2gt_5),781). orientation(lb2gt_6,151,781). coords(lb2gt_6,72,122,781).  visible(lb2gt_6,781). happensAt(inactive(lb2gt_6),781).
orientation(lb2gt_5,132,782). coords(lb2gt_5,152,39,782).  visible(lb2gt_5,782). happensAt(walking(lb2gt_5),782). orientation(lb2gt_6,151,782). coords(lb2gt_6,72,122,782).  visible(lb2gt_6,782). happensAt(inactive(lb2gt_6),782).
orientation(lb2gt_5,132,783). coords(lb2gt_5,151,40,783).  visible(lb2gt_5,783). happensAt(walking(lb2gt_5),783). orientation(lb2gt_6,151,783). coords(lb2gt_6,72,122,783).  visible(lb2gt_6,783). happensAt(inactive(lb2gt_6),783).
#show holdsAt/2.