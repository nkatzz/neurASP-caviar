
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

	orientation(wk1gt_0,131,99). coords(wk1gt_0,205,125,99).  visible(wk1gt_0,99). happensAt(walking(wk1gt_0),99). orientation(wk1gt_5,0,99). coords(wk1gt_5,35,188,99).  visible(wk1gt_5,99). happensAt(walking(wk1gt_5),99).
orientation(wk1gt_0,131,100). coords(wk1gt_0,205,125,100).  visible(wk1gt_0,100). happensAt(walking(wk1gt_0),100). orientation(wk1gt_5,0,100). coords(wk1gt_5,36,187,100).  visible(wk1gt_5,100). happensAt(walking(wk1gt_5),100).
orientation(wk1gt_0,131,101). coords(wk1gt_0,205,125,101).  visible(wk1gt_0,101). happensAt(active(wk1gt_0),101). orientation(wk1gt_5,0,101). coords(wk1gt_5,37,186,101).  visible(wk1gt_5,101). happensAt(walking(wk1gt_5),101).
orientation(wk1gt_0,131,102). coords(wk1gt_0,204,125,102).  visible(wk1gt_0,102). happensAt(active(wk1gt_0),102). orientation(wk1gt_5,0,102). coords(wk1gt_5,37,185,102).  visible(wk1gt_5,102). happensAt(walking(wk1gt_5),102).
orientation(wk1gt_0,131,103). coords(wk1gt_0,204,125,103).  visible(wk1gt_0,103). happensAt(active(wk1gt_0),103). orientation(wk1gt_5,0,103). coords(wk1gt_5,37,183,103).  visible(wk1gt_5,103). happensAt(walking(wk1gt_5),103).
orientation(wk1gt_0,131,104). coords(wk1gt_0,204,125,104).  visible(wk1gt_0,104). happensAt(active(wk1gt_0),104). orientation(wk1gt_5,0,104). coords(wk1gt_5,37,183,104).  visible(wk1gt_5,104). happensAt(walking(wk1gt_5),104).
orientation(wk1gt_0,131,105). coords(wk1gt_0,204,125,105).  visible(wk1gt_0,105). happensAt(active(wk1gt_0),105). orientation(wk1gt_5,0,105). coords(wk1gt_5,37,183,105).  visible(wk1gt_5,105). happensAt(walking(wk1gt_5),105).
orientation(wk1gt_0,131,106). coords(wk1gt_0,204,125,106).  visible(wk1gt_0,106). happensAt(active(wk1gt_0),106). orientation(wk1gt_5,0,106). coords(wk1gt_5,37,181,106).  visible(wk1gt_5,106). happensAt(walking(wk1gt_5),106).
orientation(wk1gt_0,131,107). coords(wk1gt_0,204,124,107).  visible(wk1gt_0,107). happensAt(active(wk1gt_0),107). orientation(wk1gt_5,0,107). coords(wk1gt_5,38,181,107).  visible(wk1gt_5,107). happensAt(walking(wk1gt_5),107).
orientation(wk1gt_0,131,108). coords(wk1gt_0,204,124,108).  visible(wk1gt_0,108). happensAt(active(wk1gt_0),108). orientation(wk1gt_5,0,108). coords(wk1gt_5,38,179,108).  visible(wk1gt_5,108). happensAt(walking(wk1gt_5),108).
orientation(wk1gt_0,131,109). coords(wk1gt_0,204,124,109).  visible(wk1gt_0,109). happensAt(active(wk1gt_0),109). orientation(wk1gt_5,0,109). coords(wk1gt_5,39,179,109).  visible(wk1gt_5,109). happensAt(walking(wk1gt_5),109).
orientation(wk1gt_0,131,110). coords(wk1gt_0,204,124,110).  visible(wk1gt_0,110). happensAt(active(wk1gt_0),110). orientation(wk1gt_5,0,110). coords(wk1gt_5,39,179,110).  visible(wk1gt_5,110). happensAt(walking(wk1gt_5),110).
orientation(wk1gt_0,131,111). coords(wk1gt_0,204,124,111).  visible(wk1gt_0,111). happensAt(active(wk1gt_0),111). orientation(wk1gt_5,0,111). coords(wk1gt_5,39,179,111).  visible(wk1gt_5,111). happensAt(walking(wk1gt_5),111).
orientation(wk1gt_0,131,112). coords(wk1gt_0,204,124,112).  visible(wk1gt_0,112). happensAt(inactive(wk1gt_0),112). orientation(wk1gt_5,0,112). coords(wk1gt_5,39,179,112).  visible(wk1gt_5,112). happensAt(walking(wk1gt_5),112).
orientation(wk1gt_0,131,113). coords(wk1gt_0,204,124,113).  visible(wk1gt_0,113). happensAt(inactive(wk1gt_0),113). orientation(wk1gt_5,174,113). coords(wk1gt_5,39,178,113).  visible(wk1gt_5,113). happensAt(walking(wk1gt_5),113).
orientation(wk1gt_0,131,114). coords(wk1gt_0,205,124,114).  visible(wk1gt_0,114). happensAt(inactive(wk1gt_0),114). orientation(wk1gt_5,174,114). coords(wk1gt_5,40,177,114).  visible(wk1gt_5,114). happensAt(walking(wk1gt_5),114).
orientation(wk1gt_0,131,115). coords(wk1gt_0,205,125,115).  visible(wk1gt_0,115). happensAt(inactive(wk1gt_0),115). orientation(wk1gt_5,174,115). coords(wk1gt_5,41,176,115).  visible(wk1gt_5,115). happensAt(walking(wk1gt_5),115).
orientation(wk1gt_0,131,116). coords(wk1gt_0,206,125,116).  visible(wk1gt_0,116). happensAt(inactive(wk1gt_0),116). orientation(wk1gt_5,174,116). coords(wk1gt_5,41,176,116).  visible(wk1gt_5,116). happensAt(walking(wk1gt_5),116).
orientation(wk1gt_0,131,117). coords(wk1gt_0,207,125,117).  visible(wk1gt_0,117). happensAt(inactive(wk1gt_0),117). orientation(wk1gt_5,169,117). coords(wk1gt_5,41,176,117).  visible(wk1gt_5,117). happensAt(walking(wk1gt_5),117).
orientation(wk1gt_0,131,118). coords(wk1gt_0,207,126,118).  visible(wk1gt_0,118). happensAt(active(wk1gt_0),118). orientation(wk1gt_5,169,118). coords(wk1gt_5,40,176,118).  visible(wk1gt_5,118). happensAt(walking(wk1gt_5),118).
orientation(wk1gt_0,131,119). coords(wk1gt_0,206,126,119).  visible(wk1gt_0,119). happensAt(active(wk1gt_0),119). orientation(wk1gt_5,169,119). coords(wk1gt_5,40,176,119).  visible(wk1gt_5,119). happensAt(walking(wk1gt_5),119).
orientation(wk1gt_0,131,120). coords(wk1gt_0,208,128,120).  visible(wk1gt_0,120). happensAt(active(wk1gt_0),120). orientation(wk1gt_5,169,120). coords(wk1gt_5,40,175,120).  visible(wk1gt_5,120). happensAt(walking(wk1gt_5),120).
orientation(wk1gt_0,131,121). coords(wk1gt_0,210,129,121).  visible(wk1gt_0,121). happensAt(active(wk1gt_0),121). orientation(wk1gt_5,169,121). coords(wk1gt_5,40,175,121).  visible(wk1gt_5,121). happensAt(walking(wk1gt_5),121).
orientation(wk1gt_0,131,122). coords(wk1gt_0,210,131,122).  visible(wk1gt_0,122). happensAt(active(wk1gt_0),122). orientation(wk1gt_5,169,122). coords(wk1gt_5,40,174,122).  visible(wk1gt_5,122). happensAt(walking(wk1gt_5),122).
orientation(wk1gt_0,131,123). coords(wk1gt_0,212,133,123).  visible(wk1gt_0,123). happensAt(active(wk1gt_0),123). orientation(wk1gt_5,169,123). coords(wk1gt_5,40,173,123).  visible(wk1gt_5,123). happensAt(walking(wk1gt_5),123).
#show holdsAt/2.