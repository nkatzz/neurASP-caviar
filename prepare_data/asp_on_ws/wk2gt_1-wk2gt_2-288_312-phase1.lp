
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

	orientation(wk2gt_1,125,288). coords(wk2gt_1,115,53,288).  visible(wk2gt_1,288). happensAt(inactive(wk2gt_1),288). orientation(wk2gt_2,141,288). coords(wk2gt_2,98,63,288).  visible(wk2gt_2,288). happensAt(inactive(wk2gt_2),288).
orientation(wk2gt_1,125,289). coords(wk2gt_1,115,53,289).  visible(wk2gt_1,289). happensAt(inactive(wk2gt_1),289). orientation(wk2gt_2,141,289). coords(wk2gt_2,98,63,289).  visible(wk2gt_2,289). happensAt(inactive(wk2gt_2),289).
orientation(wk2gt_1,125,290). coords(wk2gt_1,115,53,290).  visible(wk2gt_1,290). happensAt(inactive(wk2gt_1),290). orientation(wk2gt_2,141,290). coords(wk2gt_2,98,63,290).  visible(wk2gt_2,290). happensAt(inactive(wk2gt_2),290).
orientation(wk2gt_1,125,291). coords(wk2gt_1,115,53,291).  visible(wk2gt_1,291). happensAt(inactive(wk2gt_1),291). orientation(wk2gt_2,141,291). coords(wk2gt_2,98,63,291).  visible(wk2gt_2,291). happensAt(inactive(wk2gt_2),291).
orientation(wk2gt_1,125,292). coords(wk2gt_1,115,53,292).  visible(wk2gt_1,292). happensAt(inactive(wk2gt_1),292). orientation(wk2gt_2,141,292). coords(wk2gt_2,98,63,292).  visible(wk2gt_2,292). happensAt(inactive(wk2gt_2),292).
orientation(wk2gt_1,125,293). coords(wk2gt_1,115,53,293).  visible(wk2gt_1,293). happensAt(inactive(wk2gt_1),293). orientation(wk2gt_2,141,293). coords(wk2gt_2,98,63,293).  visible(wk2gt_2,293). happensAt(inactive(wk2gt_2),293).
orientation(wk2gt_1,125,294). coords(wk2gt_1,115,53,294).  visible(wk2gt_1,294). happensAt(inactive(wk2gt_1),294). orientation(wk2gt_2,141,294). coords(wk2gt_2,98,63,294).  visible(wk2gt_2,294). happensAt(inactive(wk2gt_2),294).
orientation(wk2gt_1,125,295). coords(wk2gt_1,115,53,295).  visible(wk2gt_1,295). happensAt(inactive(wk2gt_1),295). orientation(wk2gt_2,141,295). coords(wk2gt_2,98,63,295).  visible(wk2gt_2,295). happensAt(inactive(wk2gt_2),295).
orientation(wk2gt_1,125,296). coords(wk2gt_1,115,53,296).  visible(wk2gt_1,296). happensAt(inactive(wk2gt_1),296). orientation(wk2gt_2,141,296). coords(wk2gt_2,98,63,296).  visible(wk2gt_2,296). happensAt(inactive(wk2gt_2),296).
orientation(wk2gt_1,125,297). coords(wk2gt_1,115,53,297).  visible(wk2gt_1,297). happensAt(inactive(wk2gt_1),297). orientation(wk2gt_2,141,297). coords(wk2gt_2,98,63,297).  visible(wk2gt_2,297). happensAt(inactive(wk2gt_2),297).
orientation(wk2gt_1,125,298). coords(wk2gt_1,115,53,298).  visible(wk2gt_1,298). happensAt(inactive(wk2gt_1),298). orientation(wk2gt_2,141,298). coords(wk2gt_2,98,63,298).  visible(wk2gt_2,298). happensAt(inactive(wk2gt_2),298).
orientation(wk2gt_1,125,299). coords(wk2gt_1,115,53,299).  visible(wk2gt_1,299). happensAt(inactive(wk2gt_1),299). orientation(wk2gt_2,141,299). coords(wk2gt_2,98,63,299).  visible(wk2gt_2,299). happensAt(inactive(wk2gt_2),299).
orientation(wk2gt_1,125,300). coords(wk2gt_1,115,53,300).  visible(wk2gt_1,300). happensAt(inactive(wk2gt_1),300). orientation(wk2gt_2,141,300). coords(wk2gt_2,98,63,300).  visible(wk2gt_2,300). happensAt(inactive(wk2gt_2),300).
orientation(wk2gt_1,125,301). coords(wk2gt_1,115,53,301).  visible(wk2gt_1,301). happensAt(inactive(wk2gt_1),301). orientation(wk2gt_2,141,301). coords(wk2gt_2,98,63,301).  visible(wk2gt_2,301). happensAt(inactive(wk2gt_2),301).
orientation(wk2gt_1,125,302). coords(wk2gt_1,115,53,302).  visible(wk2gt_1,302). happensAt(inactive(wk2gt_1),302). orientation(wk2gt_2,141,302). coords(wk2gt_2,98,63,302).  visible(wk2gt_2,302). happensAt(inactive(wk2gt_2),302).
orientation(wk2gt_1,125,303). coords(wk2gt_1,115,53,303).  visible(wk2gt_1,303). happensAt(inactive(wk2gt_1),303). orientation(wk2gt_2,141,303). coords(wk2gt_2,98,63,303).  visible(wk2gt_2,303). happensAt(inactive(wk2gt_2),303).
orientation(wk2gt_1,125,304). coords(wk2gt_1,115,53,304).  visible(wk2gt_1,304). happensAt(inactive(wk2gt_1),304). orientation(wk2gt_2,141,304). coords(wk2gt_2,98,63,304).  visible(wk2gt_2,304). happensAt(inactive(wk2gt_2),304).
orientation(wk2gt_1,125,305). coords(wk2gt_1,115,53,305).  visible(wk2gt_1,305). happensAt(inactive(wk2gt_1),305). orientation(wk2gt_2,141,305). coords(wk2gt_2,98,63,305).  visible(wk2gt_2,305). happensAt(inactive(wk2gt_2),305).
orientation(wk2gt_1,125,306). coords(wk2gt_1,115,53,306).  visible(wk2gt_1,306). happensAt(inactive(wk2gt_1),306). orientation(wk2gt_2,141,306). coords(wk2gt_2,98,63,306).  visible(wk2gt_2,306). happensAt(inactive(wk2gt_2),306).
orientation(wk2gt_1,125,307). coords(wk2gt_1,115,53,307).  visible(wk2gt_1,307). happensAt(inactive(wk2gt_1),307). orientation(wk2gt_2,141,307). coords(wk2gt_2,98,63,307).  visible(wk2gt_2,307). happensAt(inactive(wk2gt_2),307).
orientation(wk2gt_1,125,308). coords(wk2gt_1,115,53,308).  visible(wk2gt_1,308). happensAt(inactive(wk2gt_1),308). orientation(wk2gt_2,141,308). coords(wk2gt_2,98,63,308).  visible(wk2gt_2,308). happensAt(inactive(wk2gt_2),308).
orientation(wk2gt_1,125,309). coords(wk2gt_1,115,53,309).  visible(wk2gt_1,309). happensAt(inactive(wk2gt_1),309). orientation(wk2gt_2,141,309). coords(wk2gt_2,98,63,309).  visible(wk2gt_2,309). happensAt(inactive(wk2gt_2),309).
orientation(wk2gt_1,125,310). coords(wk2gt_1,115,53,310).  visible(wk2gt_1,310). happensAt(inactive(wk2gt_1),310). orientation(wk2gt_2,141,310). coords(wk2gt_2,98,63,310).  visible(wk2gt_2,310). happensAt(inactive(wk2gt_2),310).
orientation(wk2gt_1,125,311). coords(wk2gt_1,115,53,311).  visible(wk2gt_1,311). happensAt(inactive(wk2gt_1),311). orientation(wk2gt_2,141,311). coords(wk2gt_2,98,63,311).  visible(wk2gt_2,311). happensAt(inactive(wk2gt_2),311).
orientation(wk2gt_1,125,312). coords(wk2gt_1,115,53,312).  visible(wk2gt_1,312). happensAt(inactive(wk2gt_1),312). orientation(wk2gt_2,141,312). coords(wk2gt_2,98,63,312).  visible(wk2gt_2,312). happensAt(inactive(wk2gt_2),312).
#show holdsAt/2.