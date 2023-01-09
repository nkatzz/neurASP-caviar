
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

	orientation(spgt_0,146,300). coords(spgt_0,168,121,300).  visible(spgt_0,300). happensAt(walking(spgt_0),300). orientation(spgt_1,133,300). coords(spgt_1,139,44,300).  visible(spgt_1,300). happensAt(walking(spgt_1),300).
orientation(spgt_0,147,301). coords(spgt_0,169,122,301).  visible(spgt_0,301). happensAt(walking(spgt_0),301). orientation(spgt_1,133,301). coords(spgt_1,139,44,301).  visible(spgt_1,301). happensAt(walking(spgt_1),301).
orientation(spgt_0,147,302). coords(spgt_0,169,123,302).  visible(spgt_0,302). happensAt(walking(spgt_0),302). orientation(spgt_1,133,302). coords(spgt_1,139,44,302).  visible(spgt_1,302). happensAt(walking(spgt_1),302).
orientation(spgt_0,148,303). coords(spgt_0,170,123,303).  visible(spgt_0,303). happensAt(walking(spgt_0),303). orientation(spgt_1,133,303). coords(spgt_1,140,44,303).  visible(spgt_1,303). happensAt(walking(spgt_1),303).
orientation(spgt_0,148,304). coords(spgt_0,171,124,304).  visible(spgt_0,304). happensAt(walking(spgt_0),304). orientation(spgt_1,133,304). coords(spgt_1,140,44,304).  visible(spgt_1,304). happensAt(walking(spgt_1),304).
orientation(spgt_0,149,305). coords(spgt_0,173,124,305).  visible(spgt_0,305). happensAt(walking(spgt_0),305). orientation(spgt_1,133,305). coords(spgt_1,140,44,305).  visible(spgt_1,305). happensAt(walking(spgt_1),305).
orientation(spgt_0,149,306). coords(spgt_0,175,125,306).  visible(spgt_0,306). happensAt(walking(spgt_0),306). orientation(spgt_1,133,306). coords(spgt_1,141,43,306).  visible(spgt_1,306). happensAt(walking(spgt_1),306).
orientation(spgt_0,149,307). coords(spgt_0,178,126,307).  visible(spgt_0,307). happensAt(walking(spgt_0),307). orientation(spgt_1,133,307). coords(spgt_1,142,43,307).  visible(spgt_1,307). happensAt(walking(spgt_1),307).
orientation(spgt_0,149,308). coords(spgt_0,180,128,308).  visible(spgt_0,308). happensAt(walking(spgt_0),308). orientation(spgt_1,133,308). coords(spgt_1,143,43,308).  visible(spgt_1,308). happensAt(walking(spgt_1),308).
orientation(spgt_0,149,309). coords(spgt_0,181,129,309).  visible(spgt_0,309). happensAt(walking(spgt_0),309). orientation(spgt_1,133,309). coords(spgt_1,144,42,309).  visible(spgt_1,309). happensAt(walking(spgt_1),309).
orientation(spgt_0,149,310). coords(spgt_0,182,131,310).  visible(spgt_0,310). happensAt(walking(spgt_0),310). orientation(spgt_1,133,310). coords(spgt_1,144,42,310).  visible(spgt_1,310). happensAt(walking(spgt_1),310).
orientation(spgt_0,149,311). coords(spgt_0,183,133,311).  visible(spgt_0,311). happensAt(walking(spgt_0),311). orientation(spgt_1,133,311). coords(spgt_1,145,42,311).  visible(spgt_1,311). happensAt(walking(spgt_1),311).
orientation(spgt_0,149,312). coords(spgt_0,184,133,312).  visible(spgt_0,312). happensAt(walking(spgt_0),312). orientation(spgt_1,133,312). coords(spgt_1,145,42,312).  visible(spgt_1,312). happensAt(walking(spgt_1),312).
orientation(spgt_0,149,313). coords(spgt_0,185,134,313).  visible(spgt_0,313). happensAt(walking(spgt_0),313). orientation(spgt_1,133,313). coords(spgt_1,145,41,313).  visible(spgt_1,313). happensAt(walking(spgt_1),313).
orientation(spgt_0,149,314). coords(spgt_0,186,135,314).  visible(spgt_0,314). happensAt(walking(spgt_0),314). orientation(spgt_1,133,314). coords(spgt_1,146,41,314).  visible(spgt_1,314). happensAt(walking(spgt_1),314).
orientation(spgt_0,149,315). coords(spgt_0,186,135,315).  visible(spgt_0,315). happensAt(walking(spgt_0),315). orientation(spgt_1,133,315). coords(spgt_1,147,41,315).  visible(spgt_1,315). happensAt(walking(spgt_1),315).
orientation(spgt_0,149,316). coords(spgt_0,187,136,316).  visible(spgt_0,316). happensAt(walking(spgt_0),316). orientation(spgt_1,133,316). coords(spgt_1,148,40,316).  visible(spgt_1,316). happensAt(walking(spgt_1),316).
orientation(spgt_0,149,317). coords(spgt_0,187,137,317).  visible(spgt_0,317). happensAt(walking(spgt_0),317). orientation(spgt_1,133,317). coords(spgt_1,148,40,317).  visible(spgt_1,317). happensAt(walking(spgt_1),317).
orientation(spgt_0,149,318). coords(spgt_0,187,140,318).  visible(spgt_0,318). happensAt(walking(spgt_0),318). orientation(spgt_1,133,318). coords(spgt_1,150,40,318).  visible(spgt_1,318). happensAt(walking(spgt_1),318).
orientation(spgt_0,149,319). coords(spgt_0,188,144,319).  visible(spgt_0,319). happensAt(walking(spgt_0),319). orientation(spgt_1,133,319). coords(spgt_1,150,39,319).  visible(spgt_1,319). happensAt(walking(spgt_1),319).
orientation(spgt_0,149,320). coords(spgt_0,188,146,320).  visible(spgt_0,320). happensAt(walking(spgt_0),320). orientation(spgt_1,133,320). coords(spgt_1,150,37,320).  visible(spgt_1,320). happensAt(walking(spgt_1),320).
orientation(spgt_0,149,321). coords(spgt_0,190,150,321).  visible(spgt_0,321). happensAt(walking(spgt_0),321). orientation(spgt_1,133,321). coords(spgt_1,151,37,321).  visible(spgt_1,321). happensAt(walking(spgt_1),321).
orientation(spgt_0,149,322). coords(spgt_0,191,151,322).  visible(spgt_0,322). happensAt(walking(spgt_0),322). orientation(spgt_1,133,322). coords(spgt_1,151,37,322).  visible(spgt_1,322). happensAt(walking(spgt_1),322).
orientation(spgt_0,149,323). coords(spgt_0,193,153,323).  visible(spgt_0,323). happensAt(walking(spgt_0),323). orientation(spgt_1,133,323). coords(spgt_1,151,37,323).  visible(spgt_1,323). happensAt(walking(spgt_1),323).
orientation(spgt_0,149,324). coords(spgt_0,194,155,324).  visible(spgt_0,324). happensAt(walking(spgt_0),324). orientation(spgt_1,133,324). coords(spgt_1,152,36,324).  visible(spgt_1,324). happensAt(walking(spgt_1),324).
#show holdsAt/2.