
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

	orientation(fra1gt_6,128,294). coords(fra1gt_6,185,75,294).  visible(fra1gt_6,294). happensAt(walking(fra1gt_6),294). orientation(fra1gt_7,142,294). coords(fra1gt_7,137,51,294).  visible(fra1gt_7,294). happensAt(walking(fra1gt_7),294).
orientation(fra1gt_6,128,295). coords(fra1gt_6,185,75,295).  visible(fra1gt_6,295). happensAt(walking(fra1gt_6),295). orientation(fra1gt_7,142,295). coords(fra1gt_7,137,52,295).  visible(fra1gt_7,295). happensAt(walking(fra1gt_7),295).
orientation(fra1gt_6,128,296). coords(fra1gt_6,184,75,296).  visible(fra1gt_6,296). happensAt(walking(fra1gt_6),296). orientation(fra1gt_7,142,296). coords(fra1gt_7,137,52,296).  visible(fra1gt_7,296). happensAt(walking(fra1gt_7),296).
orientation(fra1gt_6,128,297). coords(fra1gt_6,183,75,297).  visible(fra1gt_6,297). happensAt(walking(fra1gt_6),297). orientation(fra1gt_7,142,297). coords(fra1gt_7,137,52,297).  visible(fra1gt_7,297). happensAt(walking(fra1gt_7),297).
orientation(fra1gt_6,128,298). coords(fra1gt_6,182,76,298).  visible(fra1gt_6,298). happensAt(walking(fra1gt_6),298). orientation(fra1gt_7,142,298). coords(fra1gt_7,137,52,298).  visible(fra1gt_7,298). happensAt(walking(fra1gt_7),298).
orientation(fra1gt_6,128,299). coords(fra1gt_6,182,76,299).  visible(fra1gt_6,299). happensAt(walking(fra1gt_6),299). orientation(fra1gt_7,142,299). coords(fra1gt_7,137,53,299).  visible(fra1gt_7,299). happensAt(walking(fra1gt_7),299).
orientation(fra1gt_6,128,300). coords(fra1gt_6,182,76,300).  visible(fra1gt_6,300). happensAt(walking(fra1gt_6),300). orientation(fra1gt_7,142,300). coords(fra1gt_7,137,55,300).  visible(fra1gt_7,300). happensAt(walking(fra1gt_7),300).
orientation(fra1gt_6,128,301). coords(fra1gt_6,181,76,301).  visible(fra1gt_6,301). happensAt(walking(fra1gt_6),301). orientation(fra1gt_7,142,301). coords(fra1gt_7,138,57,301).  visible(fra1gt_7,301). happensAt(walking(fra1gt_7),301).
orientation(fra1gt_6,131,302). coords(fra1gt_6,181,76,302).  visible(fra1gt_6,302). happensAt(walking(fra1gt_6),302). orientation(fra1gt_7,142,302). coords(fra1gt_7,137,59,302).  visible(fra1gt_7,302). happensAt(walking(fra1gt_7),302).
orientation(fra1gt_6,133,303). coords(fra1gt_6,180,76,303).  visible(fra1gt_6,303). happensAt(walking(fra1gt_6),303). orientation(fra1gt_7,142,303). coords(fra1gt_7,138,59,303).  visible(fra1gt_7,303). happensAt(walking(fra1gt_7),303).
orientation(fra1gt_6,133,304). coords(fra1gt_6,180,76,304).  visible(fra1gt_6,304). happensAt(walking(fra1gt_6),304). orientation(fra1gt_7,142,304). coords(fra1gt_7,138,60,304).  visible(fra1gt_7,304). happensAt(walking(fra1gt_7),304).
orientation(fra1gt_6,133,305). coords(fra1gt_6,177,76,305).  visible(fra1gt_6,305). happensAt(walking(fra1gt_6),305). orientation(fra1gt_7,142,305). coords(fra1gt_7,138,62,305).  visible(fra1gt_7,305). happensAt(walking(fra1gt_7),305).
orientation(fra1gt_6,133,306). coords(fra1gt_6,175,76,306).  visible(fra1gt_6,306). happensAt(walking(fra1gt_6),306). orientation(fra1gt_7,142,306). coords(fra1gt_7,138,62,306).  visible(fra1gt_7,306). happensAt(walking(fra1gt_7),306).
orientation(fra1gt_6,133,307). coords(fra1gt_6,172,76,307).  visible(fra1gt_6,307). happensAt(walking(fra1gt_6),307). orientation(fra1gt_7,142,307). coords(fra1gt_7,138,62,307).  visible(fra1gt_7,307). happensAt(walking(fra1gt_7),307).
orientation(fra1gt_6,133,308). coords(fra1gt_6,171,77,308).  visible(fra1gt_6,308). happensAt(walking(fra1gt_6),308). orientation(fra1gt_7,142,308). coords(fra1gt_7,138,62,308).  visible(fra1gt_7,308). happensAt(walking(fra1gt_7),308).
orientation(fra1gt_6,133,309). coords(fra1gt_6,169,78,309).  visible(fra1gt_6,309). happensAt(walking(fra1gt_6),309). orientation(fra1gt_7,142,309). coords(fra1gt_7,138,63,309).  visible(fra1gt_7,309). happensAt(walking(fra1gt_7),309).
orientation(fra1gt_6,133,310). coords(fra1gt_6,168,78,310).  visible(fra1gt_6,310). happensAt(walking(fra1gt_6),310). orientation(fra1gt_7,142,310). coords(fra1gt_7,138,63,310).  visible(fra1gt_7,310). happensAt(walking(fra1gt_7),310).
orientation(fra1gt_6,133,311). coords(fra1gt_6,167,78,311).  visible(fra1gt_6,311). happensAt(walking(fra1gt_6),311). orientation(fra1gt_7,142,311). coords(fra1gt_7,139,63,311).  visible(fra1gt_7,311). happensAt(walking(fra1gt_7),311).
orientation(fra1gt_6,133,312). coords(fra1gt_6,167,77,312).  visible(fra1gt_6,312). happensAt(walking(fra1gt_6),312). orientation(fra1gt_7,142,312). coords(fra1gt_7,139,64,312).  visible(fra1gt_7,312). happensAt(walking(fra1gt_7),312).
orientation(fra1gt_6,133,313). coords(fra1gt_6,166,78,313).  visible(fra1gt_6,313). happensAt(walking(fra1gt_6),313). orientation(fra1gt_7,142,313). coords(fra1gt_7,140,64,313).  visible(fra1gt_7,313). happensAt(walking(fra1gt_7),313).
orientation(fra1gt_6,133,314). coords(fra1gt_6,166,78,314).  visible(fra1gt_6,314). happensAt(walking(fra1gt_6),314). orientation(fra1gt_7,142,314). coords(fra1gt_7,140,64,314).  visible(fra1gt_7,314). happensAt(walking(fra1gt_7),314).
orientation(fra1gt_6,133,315). coords(fra1gt_6,165,79,315).  visible(fra1gt_6,315). happensAt(walking(fra1gt_6),315). orientation(fra1gt_7,142,315). coords(fra1gt_7,140,64,315).  visible(fra1gt_7,315). happensAt(walking(fra1gt_7),315).
orientation(fra1gt_6,137,316). coords(fra1gt_6,165,79,316).  visible(fra1gt_6,316). happensAt(walking(fra1gt_6),316). orientation(fra1gt_7,142,316). coords(fra1gt_7,140,64,316).  visible(fra1gt_7,316). happensAt(walking(fra1gt_7),316).
orientation(fra1gt_6,137,317). coords(fra1gt_6,164,79,317).  visible(fra1gt_6,317). happensAt(walking(fra1gt_6),317). orientation(fra1gt_7,142,317). coords(fra1gt_7,141,65,317).  visible(fra1gt_7,317). happensAt(walking(fra1gt_7),317).
orientation(fra1gt_6,137,318). coords(fra1gt_6,162,78,318).  visible(fra1gt_6,318). happensAt(walking(fra1gt_6),318). orientation(fra1gt_7,142,318). coords(fra1gt_7,141,65,318).  visible(fra1gt_7,318). happensAt(walking(fra1gt_7),318).
#show holdsAt/2.