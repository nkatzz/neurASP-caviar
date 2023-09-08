
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

	orientation(wk2gt_1,125,756). coords(wk2gt_1,115,53,756).  visible(wk2gt_1,756). happensAt(inactive(wk2gt_1),756). orientation(wk2gt_2,142,756). coords(wk2gt_2,99,63,756).  visible(wk2gt_2,756). happensAt(inactive(wk2gt_2),756).
orientation(wk2gt_1,125,757). coords(wk2gt_1,115,53,757).  visible(wk2gt_1,757). happensAt(inactive(wk2gt_1),757). orientation(wk2gt_2,142,757). coords(wk2gt_2,99,63,757).  visible(wk2gt_2,757). happensAt(inactive(wk2gt_2),757).
orientation(wk2gt_1,125,758). coords(wk2gt_1,115,53,758).  visible(wk2gt_1,758). happensAt(inactive(wk2gt_1),758). orientation(wk2gt_2,142,758). coords(wk2gt_2,99,63,758).  visible(wk2gt_2,758). happensAt(inactive(wk2gt_2),758).
orientation(wk2gt_1,125,759). coords(wk2gt_1,115,53,759).  visible(wk2gt_1,759). happensAt(inactive(wk2gt_1),759). orientation(wk2gt_2,142,759). coords(wk2gt_2,99,63,759).  visible(wk2gt_2,759). happensAt(inactive(wk2gt_2),759).
orientation(wk2gt_1,125,760). coords(wk2gt_1,115,53,760).  visible(wk2gt_1,760). happensAt(inactive(wk2gt_1),760). orientation(wk2gt_2,142,760). coords(wk2gt_2,99,63,760).  visible(wk2gt_2,760). happensAt(inactive(wk2gt_2),760).
orientation(wk2gt_1,125,761). coords(wk2gt_1,115,53,761).  visible(wk2gt_1,761). happensAt(inactive(wk2gt_1),761). orientation(wk2gt_2,142,761). coords(wk2gt_2,99,63,761).  visible(wk2gt_2,761). happensAt(inactive(wk2gt_2),761).
orientation(wk2gt_1,125,762). coords(wk2gt_1,115,53,762).  visible(wk2gt_1,762). happensAt(inactive(wk2gt_1),762). orientation(wk2gt_2,142,762). coords(wk2gt_2,99,63,762).  visible(wk2gt_2,762). happensAt(inactive(wk2gt_2),762).
orientation(wk2gt_1,125,763). coords(wk2gt_1,115,53,763).  visible(wk2gt_1,763). happensAt(inactive(wk2gt_1),763). orientation(wk2gt_2,142,763). coords(wk2gt_2,99,63,763).  visible(wk2gt_2,763). happensAt(inactive(wk2gt_2),763).
orientation(wk2gt_1,125,764). coords(wk2gt_1,115,53,764).  visible(wk2gt_1,764). happensAt(inactive(wk2gt_1),764). orientation(wk2gt_2,142,764). coords(wk2gt_2,99,63,764).  visible(wk2gt_2,764). happensAt(inactive(wk2gt_2),764).
orientation(wk2gt_1,125,765). coords(wk2gt_1,115,53,765).  visible(wk2gt_1,765). happensAt(inactive(wk2gt_1),765). orientation(wk2gt_2,142,765). coords(wk2gt_2,99,63,765).  visible(wk2gt_2,765). happensAt(inactive(wk2gt_2),765).
orientation(wk2gt_1,125,766). coords(wk2gt_1,115,53,766).  visible(wk2gt_1,766). happensAt(inactive(wk2gt_1),766). orientation(wk2gt_2,142,766). coords(wk2gt_2,99,63,766).  visible(wk2gt_2,766). happensAt(inactive(wk2gt_2),766).
orientation(wk2gt_1,125,767). coords(wk2gt_1,115,53,767).  visible(wk2gt_1,767). happensAt(inactive(wk2gt_1),767). orientation(wk2gt_2,142,767). coords(wk2gt_2,99,63,767).  visible(wk2gt_2,767). happensAt(inactive(wk2gt_2),767).
orientation(wk2gt_1,125,768). coords(wk2gt_1,115,53,768).  visible(wk2gt_1,768). happensAt(inactive(wk2gt_1),768). orientation(wk2gt_2,142,768). coords(wk2gt_2,99,63,768).  visible(wk2gt_2,768). happensAt(inactive(wk2gt_2),768).
orientation(wk2gt_1,125,769). coords(wk2gt_1,115,53,769).  visible(wk2gt_1,769). happensAt(inactive(wk2gt_1),769). orientation(wk2gt_2,142,769). coords(wk2gt_2,99,63,769).  visible(wk2gt_2,769). happensAt(inactive(wk2gt_2),769).
orientation(wk2gt_1,125,770). coords(wk2gt_1,115,53,770).  visible(wk2gt_1,770). happensAt(inactive(wk2gt_1),770). orientation(wk2gt_2,142,770). coords(wk2gt_2,99,63,770).  visible(wk2gt_2,770). happensAt(inactive(wk2gt_2),770).
orientation(wk2gt_1,125,771). coords(wk2gt_1,115,53,771).  visible(wk2gt_1,771). happensAt(inactive(wk2gt_1),771). orientation(wk2gt_2,142,771). coords(wk2gt_2,99,63,771).  visible(wk2gt_2,771). happensAt(inactive(wk2gt_2),771).
orientation(wk2gt_1,125,772). coords(wk2gt_1,115,53,772).  visible(wk2gt_1,772). happensAt(inactive(wk2gt_1),772). orientation(wk2gt_2,142,772). coords(wk2gt_2,99,63,772).  visible(wk2gt_2,772). happensAt(inactive(wk2gt_2),772).
orientation(wk2gt_1,125,773). coords(wk2gt_1,115,53,773).  visible(wk2gt_1,773). happensAt(inactive(wk2gt_1),773). orientation(wk2gt_2,142,773). coords(wk2gt_2,99,63,773).  visible(wk2gt_2,773). happensAt(inactive(wk2gt_2),773).
orientation(wk2gt_1,125,774). coords(wk2gt_1,115,53,774).  visible(wk2gt_1,774). happensAt(inactive(wk2gt_1),774). orientation(wk2gt_2,142,774). coords(wk2gt_2,99,63,774).  visible(wk2gt_2,774). happensAt(inactive(wk2gt_2),774).
orientation(wk2gt_1,125,775). coords(wk2gt_1,115,53,775).  visible(wk2gt_1,775). happensAt(inactive(wk2gt_1),775). orientation(wk2gt_2,142,775). coords(wk2gt_2,99,63,775).  visible(wk2gt_2,775). happensAt(inactive(wk2gt_2),775).
orientation(wk2gt_1,125,776). coords(wk2gt_1,115,53,776).  visible(wk2gt_1,776). happensAt(inactive(wk2gt_1),776). orientation(wk2gt_2,142,776). coords(wk2gt_2,99,63,776).  visible(wk2gt_2,776). happensAt(inactive(wk2gt_2),776).
orientation(wk2gt_1,125,777). coords(wk2gt_1,115,53,777).  visible(wk2gt_1,777). happensAt(inactive(wk2gt_1),777). orientation(wk2gt_2,142,777). coords(wk2gt_2,99,63,777).  visible(wk2gt_2,777). happensAt(inactive(wk2gt_2),777).
orientation(wk2gt_1,125,778). coords(wk2gt_1,115,53,778).  visible(wk2gt_1,778). happensAt(inactive(wk2gt_1),778). orientation(wk2gt_2,142,778). coords(wk2gt_2,99,63,778).  visible(wk2gt_2,778). happensAt(inactive(wk2gt_2),778).
orientation(wk2gt_1,125,779). coords(wk2gt_1,115,53,779).  visible(wk2gt_1,779). happensAt(inactive(wk2gt_1),779). orientation(wk2gt_2,142,779). coords(wk2gt_2,99,63,779).  visible(wk2gt_2,779). happensAt(inactive(wk2gt_2),779).
orientation(wk2gt_1,125,780). coords(wk2gt_1,115,53,780).  visible(wk2gt_1,780). happensAt(inactive(wk2gt_1),780). orientation(wk2gt_2,142,780). coords(wk2gt_2,99,63,780).  visible(wk2gt_2,780). happensAt(inactive(wk2gt_2),780).
#show holdsAt/2.