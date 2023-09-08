
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

	orientation(bww1gt_1,7,290). coords(bww1gt_1,101,262,290).  visible(bww1gt_1,290). happensAt(inactive(bww1gt_1),290). orientation(bww1gt_2,167,290). coords(bww1gt_2,54,164,290).  visible(bww1gt_2,290). happensAt(walking(bww1gt_2),290).
orientation(bww1gt_1,3,291). coords(bww1gt_1,101,262,291).  visible(bww1gt_1,291). happensAt(inactive(bww1gt_1),291). orientation(bww1gt_2,167,291). coords(bww1gt_2,55,164,291).  visible(bww1gt_2,291). happensAt(walking(bww1gt_2),291).
orientation(bww1gt_1,7,292). coords(bww1gt_1,102,262,292).  visible(bww1gt_1,292). happensAt(inactive(bww1gt_1),292). orientation(bww1gt_2,167,292). coords(bww1gt_2,56,164,292).  visible(bww1gt_2,292). happensAt(walking(bww1gt_2),292).
orientation(bww1gt_1,4,293). coords(bww1gt_1,102,262,293).  visible(bww1gt_1,293). happensAt(inactive(bww1gt_1),293). orientation(bww1gt_2,167,293). coords(bww1gt_2,56,164,293).  visible(bww1gt_2,293). happensAt(walking(bww1gt_2),293).
orientation(bww1gt_1,4,294). coords(bww1gt_1,102,262,294).  visible(bww1gt_1,294). happensAt(inactive(bww1gt_1),294). orientation(bww1gt_2,167,294). coords(bww1gt_2,56,164,294).  visible(bww1gt_2,294). happensAt(walking(bww1gt_2),294).
orientation(bww1gt_1,7,295). coords(bww1gt_1,103,263,295).  visible(bww1gt_1,295). happensAt(inactive(bww1gt_1),295). orientation(bww1gt_2,167,295). coords(bww1gt_2,57,164,295).  visible(bww1gt_2,295). happensAt(walking(bww1gt_2),295).
orientation(bww1gt_1,4,296). coords(bww1gt_1,103,263,296).  visible(bww1gt_1,296). happensAt(inactive(bww1gt_1),296). orientation(bww1gt_2,167,296). coords(bww1gt_2,57,163,296).  visible(bww1gt_2,296). happensAt(walking(bww1gt_2),296).
orientation(bww1gt_1,8,297). coords(bww1gt_1,103,263,297).  visible(bww1gt_1,297). happensAt(inactive(bww1gt_1),297). orientation(bww1gt_2,167,297). coords(bww1gt_2,57,163,297).  visible(bww1gt_2,297). happensAt(walking(bww1gt_2),297).
orientation(bww1gt_1,179,298). coords(bww1gt_1,103,263,298).  visible(bww1gt_1,298). happensAt(inactive(bww1gt_1),298). orientation(bww1gt_2,167,298). coords(bww1gt_2,57,163,298).  visible(bww1gt_2,298). happensAt(walking(bww1gt_2),298).
orientation(bww1gt_1,0,299). coords(bww1gt_1,103,263,299).  visible(bww1gt_1,299). happensAt(inactive(bww1gt_1),299). orientation(bww1gt_2,167,299). coords(bww1gt_2,57,163,299).  visible(bww1gt_2,299). happensAt(walking(bww1gt_2),299).
orientation(bww1gt_1,3,300). coords(bww1gt_1,103,263,300).  visible(bww1gt_1,300). happensAt(inactive(bww1gt_1),300). orientation(bww1gt_2,167,300). coords(bww1gt_2,57,162,300).  visible(bww1gt_2,300). happensAt(walking(bww1gt_2),300).
orientation(bww1gt_1,6,301). coords(bww1gt_1,104,264,301).  visible(bww1gt_1,301). happensAt(inactive(bww1gt_1),301). orientation(bww1gt_2,167,301). coords(bww1gt_2,57,162,301).  visible(bww1gt_2,301). happensAt(walking(bww1gt_2),301).
orientation(bww1gt_1,3,302). coords(bww1gt_1,104,264,302).  visible(bww1gt_1,302). happensAt(inactive(bww1gt_1),302). orientation(bww1gt_2,167,302). coords(bww1gt_2,57,162,302).  visible(bww1gt_2,302). happensAt(walking(bww1gt_2),302).
orientation(bww1gt_1,176,303). coords(bww1gt_1,104,264,303).  visible(bww1gt_1,303). happensAt(inactive(bww1gt_1),303). orientation(bww1gt_2,167,303). coords(bww1gt_2,58,162,303).  visible(bww1gt_2,303). happensAt(walking(bww1gt_2),303).
orientation(bww1gt_1,175,304). coords(bww1gt_1,104,264,304).  visible(bww1gt_1,304). happensAt(inactive(bww1gt_1),304). orientation(bww1gt_2,167,304). coords(bww1gt_2,58,162,304).  visible(bww1gt_2,304). happensAt(walking(bww1gt_2),304).
orientation(bww1gt_1,177,305). coords(bww1gt_1,104,264,305).  visible(bww1gt_1,305). happensAt(inactive(bww1gt_1),305). orientation(bww1gt_2,167,305). coords(bww1gt_2,59,162,305).  visible(bww1gt_2,305). happensAt(walking(bww1gt_2),305).
orientation(bww1gt_1,179,306). coords(bww1gt_1,104,264,306).  visible(bww1gt_1,306). happensAt(inactive(bww1gt_1),306). orientation(bww1gt_2,167,306). coords(bww1gt_2,59,162,306).  visible(bww1gt_2,306). happensAt(walking(bww1gt_2),306).
orientation(bww1gt_1,2,307). coords(bww1gt_1,104,264,307).  visible(bww1gt_1,307). happensAt(inactive(bww1gt_1),307). orientation(bww1gt_2,167,307). coords(bww1gt_2,60,161,307).  visible(bww1gt_2,307). happensAt(walking(bww1gt_2),307).
orientation(bww1gt_1,2,308). coords(bww1gt_1,104,264,308).  visible(bww1gt_1,308). happensAt(inactive(bww1gt_1),308). orientation(bww1gt_2,167,308). coords(bww1gt_2,61,161,308).  visible(bww1gt_2,308). happensAt(walking(bww1gt_2),308).
orientation(bww1gt_1,1,309). coords(bww1gt_1,104,264,309).  visible(bww1gt_1,309). happensAt(inactive(bww1gt_1),309). orientation(bww1gt_2,167,309). coords(bww1gt_2,62,161,309).  visible(bww1gt_2,309). happensAt(walking(bww1gt_2),309).
orientation(bww1gt_1,3,310). coords(bww1gt_1,104,264,310).  visible(bww1gt_1,310). happensAt(inactive(bww1gt_1),310). orientation(bww1gt_2,167,310). coords(bww1gt_2,62,161,310).  visible(bww1gt_2,310). happensAt(walking(bww1gt_2),310).
orientation(bww1gt_1,2,311). coords(bww1gt_1,104,264,311).  visible(bww1gt_1,311). happensAt(inactive(bww1gt_1),311). orientation(bww1gt_2,167,311). coords(bww1gt_2,62,161,311).  visible(bww1gt_2,311). happensAt(walking(bww1gt_2),311).
orientation(bww1gt_1,6,312). coords(bww1gt_1,105,264,312).  visible(bww1gt_1,312). happensAt(inactive(bww1gt_1),312). orientation(bww1gt_2,167,312). coords(bww1gt_2,62,161,312).  visible(bww1gt_2,312). happensAt(walking(bww1gt_2),312).
orientation(bww1gt_1,5,313). coords(bww1gt_1,105,264,313).  visible(bww1gt_1,313). happensAt(inactive(bww1gt_1),313). orientation(bww1gt_2,167,313). coords(bww1gt_2,63,161,313).  visible(bww1gt_2,313). happensAt(walking(bww1gt_2),313).
orientation(bww1gt_1,7,314). coords(bww1gt_1,105,264,314).  visible(bww1gt_1,314). happensAt(inactive(bww1gt_1),314). orientation(bww1gt_2,167,314). coords(bww1gt_2,63,161,314).  visible(bww1gt_2,314). happensAt(walking(bww1gt_2),314).
#show holdsAt/2.