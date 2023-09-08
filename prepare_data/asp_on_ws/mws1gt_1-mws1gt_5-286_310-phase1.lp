
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

	orientation(mws1gt_1,155,286). coords(mws1gt_1,91,106,286).  visible(mws1gt_1,286). happensAt(walking(mws1gt_1),286). orientation(mws1gt_5,161,286). coords(mws1gt_5,37,165,286).  visible(mws1gt_5,286). happensAt(walking(mws1gt_5),286).
orientation(mws1gt_1,155,287). coords(mws1gt_1,89,106,287).  visible(mws1gt_1,287). happensAt(walking(mws1gt_1),287). orientation(mws1gt_5,161,287). coords(mws1gt_5,37,166,287).  visible(mws1gt_5,287). happensAt(walking(mws1gt_5),287).
orientation(mws1gt_1,155,288). coords(mws1gt_1,88,105,288).  visible(mws1gt_1,288). happensAt(walking(mws1gt_1),288). orientation(mws1gt_5,161,288). coords(mws1gt_5,37,166,288).  visible(mws1gt_5,288). happensAt(walking(mws1gt_5),288).
orientation(mws1gt_1,155,289). coords(mws1gt_1,88,105,289).  visible(mws1gt_1,289). happensAt(walking(mws1gt_1),289). orientation(mws1gt_5,161,289). coords(mws1gt_5,37,167,289).  visible(mws1gt_5,289). happensAt(walking(mws1gt_5),289).
orientation(mws1gt_1,155,290). coords(mws1gt_1,87,104,290).  visible(mws1gt_1,290). happensAt(walking(mws1gt_1),290). orientation(mws1gt_5,161,290). coords(mws1gt_5,37,168,290).  visible(mws1gt_5,290). happensAt(walking(mws1gt_5),290).
orientation(mws1gt_1,155,291). coords(mws1gt_1,87,104,291).  visible(mws1gt_1,291). happensAt(walking(mws1gt_1),291). orientation(mws1gt_5,168,291). coords(mws1gt_5,37,169,291).  visible(mws1gt_5,291). happensAt(walking(mws1gt_5),291).
orientation(mws1gt_1,155,292). coords(mws1gt_1,87,104,292).  visible(mws1gt_1,292). happensAt(walking(mws1gt_1),292). orientation(mws1gt_5,168,292). coords(mws1gt_5,37,170,292).  visible(mws1gt_5,292). happensAt(walking(mws1gt_5),292).
orientation(mws1gt_1,155,293). coords(mws1gt_1,87,104,293).  visible(mws1gt_1,293). happensAt(walking(mws1gt_1),293). orientation(mws1gt_5,168,293). coords(mws1gt_5,37,171,293).  visible(mws1gt_5,293). happensAt(walking(mws1gt_5),293).
orientation(mws1gt_1,155,294). coords(mws1gt_1,87,103,294).  visible(mws1gt_1,294). happensAt(walking(mws1gt_1),294). orientation(mws1gt_5,168,294). coords(mws1gt_5,37,172,294).  visible(mws1gt_5,294). happensAt(walking(mws1gt_5),294).
orientation(mws1gt_1,155,295). coords(mws1gt_1,87,103,295).  visible(mws1gt_1,295). happensAt(walking(mws1gt_1),295). orientation(mws1gt_5,168,295). coords(mws1gt_5,37,174,295).  visible(mws1gt_5,295). happensAt(walking(mws1gt_5),295).
orientation(mws1gt_1,155,296). coords(mws1gt_1,87,103,296).  visible(mws1gt_1,296). happensAt(walking(mws1gt_1),296). orientation(mws1gt_5,168,296). coords(mws1gt_5,36,175,296).  visible(mws1gt_5,296). happensAt(walking(mws1gt_5),296).
orientation(mws1gt_1,155,297). coords(mws1gt_1,86,103,297).  visible(mws1gt_1,297). happensAt(walking(mws1gt_1),297). orientation(mws1gt_5,168,297). coords(mws1gt_5,36,176,297).  visible(mws1gt_5,297). happensAt(walking(mws1gt_5),297).
orientation(mws1gt_1,155,298). coords(mws1gt_1,86,103,298).  visible(mws1gt_1,298). happensAt(walking(mws1gt_1),298). orientation(mws1gt_5,168,298). coords(mws1gt_5,36,177,298).  visible(mws1gt_5,298). happensAt(walking(mws1gt_5),298).
orientation(mws1gt_1,155,299). coords(mws1gt_1,85,104,299).  visible(mws1gt_1,299). happensAt(walking(mws1gt_1),299). orientation(mws1gt_5,168,299). coords(mws1gt_5,36,178,299).  visible(mws1gt_5,299). happensAt(walking(mws1gt_5),299).
orientation(mws1gt_1,155,300). coords(mws1gt_1,85,104,300).  visible(mws1gt_1,300). happensAt(walking(mws1gt_1),300). orientation(mws1gt_5,168,300). coords(mws1gt_5,36,178,300).  visible(mws1gt_5,300). happensAt(walking(mws1gt_5),300).
orientation(mws1gt_1,155,301). coords(mws1gt_1,84,103,301).  visible(mws1gt_1,301). happensAt(walking(mws1gt_1),301). orientation(mws1gt_5,168,301). coords(mws1gt_5,36,179,301).  visible(mws1gt_5,301). happensAt(walking(mws1gt_5),301).
orientation(mws1gt_1,155,302). coords(mws1gt_1,83,103,302).  visible(mws1gt_1,302). happensAt(walking(mws1gt_1),302). orientation(mws1gt_5,168,302). coords(mws1gt_5,36,179,302).  visible(mws1gt_5,302). happensAt(walking(mws1gt_5),302).
orientation(mws1gt_1,155,303). coords(mws1gt_1,82,103,303).  visible(mws1gt_1,303). happensAt(walking(mws1gt_1),303). orientation(mws1gt_5,168,303). coords(mws1gt_5,36,179,303).  visible(mws1gt_5,303). happensAt(walking(mws1gt_5),303).
orientation(mws1gt_1,155,304). coords(mws1gt_1,82,103,304).  visible(mws1gt_1,304). happensAt(walking(mws1gt_1),304). orientation(mws1gt_5,168,304). coords(mws1gt_5,36,179,304).  visible(mws1gt_5,304). happensAt(walking(mws1gt_5),304).
orientation(mws1gt_1,155,305). coords(mws1gt_1,81,103,305).  visible(mws1gt_1,305). happensAt(walking(mws1gt_1),305). orientation(mws1gt_5,168,305). coords(mws1gt_5,36,180,305).  visible(mws1gt_5,305). happensAt(walking(mws1gt_5),305).
orientation(mws1gt_1,155,306). coords(mws1gt_1,81,103,306).  visible(mws1gt_1,306). happensAt(walking(mws1gt_1),306). orientation(mws1gt_5,168,306). coords(mws1gt_5,36,181,306).  visible(mws1gt_5,306). happensAt(walking(mws1gt_5),306).
orientation(mws1gt_1,155,307). coords(mws1gt_1,80,103,307).  visible(mws1gt_1,307). happensAt(walking(mws1gt_1),307). orientation(mws1gt_5,168,307). coords(mws1gt_5,36,182,307).  visible(mws1gt_5,307). happensAt(walking(mws1gt_5),307).
orientation(mws1gt_1,155,308). coords(mws1gt_1,80,103,308).  visible(mws1gt_1,308). happensAt(walking(mws1gt_1),308). orientation(mws1gt_5,168,308). coords(mws1gt_5,35,183,308).  visible(mws1gt_5,308). happensAt(walking(mws1gt_5),308).
orientation(mws1gt_1,155,309). coords(mws1gt_1,80,103,309).  visible(mws1gt_1,309). happensAt(walking(mws1gt_1),309). orientation(mws1gt_5,168,309). coords(mws1gt_5,35,185,309).  visible(mws1gt_5,309). happensAt(walking(mws1gt_5),309).
orientation(mws1gt_1,155,310). coords(mws1gt_1,80,103,310).  visible(mws1gt_1,310). happensAt(walking(mws1gt_1),310). orientation(mws1gt_5,168,310). coords(mws1gt_5,35,186,310).  visible(mws1gt_5,310). happensAt(walking(mws1gt_5),310).
#show holdsAt/2.