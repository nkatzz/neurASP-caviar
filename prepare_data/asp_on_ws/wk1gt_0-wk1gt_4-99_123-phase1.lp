
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

	orientation(wk1gt_0,131,99). coords(wk1gt_0,205,125,99).  visible(wk1gt_0,99). happensAt(walking(wk1gt_0),99). orientation(wk1gt_4,0,99). coords(wk1gt_4,41,193,99).  visible(wk1gt_4,99). happensAt(walking(wk1gt_4),99).
orientation(wk1gt_0,131,100). coords(wk1gt_0,205,125,100).  visible(wk1gt_0,100). happensAt(walking(wk1gt_0),100). orientation(wk1gt_4,0,100). coords(wk1gt_4,41,193,100).  visible(wk1gt_4,100). happensAt(walking(wk1gt_4),100).
orientation(wk1gt_0,131,101). coords(wk1gt_0,205,125,101).  visible(wk1gt_0,101). happensAt(active(wk1gt_0),101). orientation(wk1gt_4,0,101). coords(wk1gt_4,43,191,101).  visible(wk1gt_4,101). happensAt(walking(wk1gt_4),101).
orientation(wk1gt_0,131,102). coords(wk1gt_0,204,125,102).  visible(wk1gt_0,102). happensAt(active(wk1gt_0),102). orientation(wk1gt_4,0,102). coords(wk1gt_4,44,190,102).  visible(wk1gt_4,102). happensAt(walking(wk1gt_4),102).
orientation(wk1gt_0,131,103). coords(wk1gt_0,204,125,103).  visible(wk1gt_0,103). happensAt(active(wk1gt_0),103). orientation(wk1gt_4,0,103). coords(wk1gt_4,44,190,103).  visible(wk1gt_4,103). happensAt(walking(wk1gt_4),103).
orientation(wk1gt_0,131,104). coords(wk1gt_0,204,125,104).  visible(wk1gt_0,104). happensAt(active(wk1gt_0),104). orientation(wk1gt_4,0,104). coords(wk1gt_4,44,190,104).  visible(wk1gt_4,104). happensAt(walking(wk1gt_4),104).
orientation(wk1gt_0,131,105). coords(wk1gt_0,204,125,105).  visible(wk1gt_0,105). happensAt(active(wk1gt_0),105). orientation(wk1gt_4,0,105). coords(wk1gt_4,44,189,105).  visible(wk1gt_4,105). happensAt(walking(wk1gt_4),105).
orientation(wk1gt_0,131,106). coords(wk1gt_0,204,125,106).  visible(wk1gt_0,106). happensAt(active(wk1gt_0),106). orientation(wk1gt_4,0,106). coords(wk1gt_4,44,189,106).  visible(wk1gt_4,106). happensAt(walking(wk1gt_4),106).
orientation(wk1gt_0,131,107). coords(wk1gt_0,204,124,107).  visible(wk1gt_0,107). happensAt(active(wk1gt_0),107). orientation(wk1gt_4,0,107). coords(wk1gt_4,44,188,107).  visible(wk1gt_4,107). happensAt(walking(wk1gt_4),107).
orientation(wk1gt_0,131,108). coords(wk1gt_0,204,124,108).  visible(wk1gt_0,108). happensAt(active(wk1gt_0),108). orientation(wk1gt_4,0,108). coords(wk1gt_4,44,187,108).  visible(wk1gt_4,108). happensAt(walking(wk1gt_4),108).
orientation(wk1gt_0,131,109). coords(wk1gt_0,204,124,109).  visible(wk1gt_0,109). happensAt(active(wk1gt_0),109). orientation(wk1gt_4,0,109). coords(wk1gt_4,44,187,109).  visible(wk1gt_4,109). happensAt(walking(wk1gt_4),109).
orientation(wk1gt_0,131,110). coords(wk1gt_0,204,124,110).  visible(wk1gt_0,110). happensAt(active(wk1gt_0),110). orientation(wk1gt_4,0,110). coords(wk1gt_4,44,187,110).  visible(wk1gt_4,110). happensAt(walking(wk1gt_4),110).
orientation(wk1gt_0,131,111). coords(wk1gt_0,204,124,111).  visible(wk1gt_0,111). happensAt(active(wk1gt_0),111). orientation(wk1gt_4,0,111). coords(wk1gt_4,44,187,111).  visible(wk1gt_4,111). happensAt(walking(wk1gt_4),111).
orientation(wk1gt_0,131,112). coords(wk1gt_0,204,124,112).  visible(wk1gt_0,112). happensAt(inactive(wk1gt_0),112). orientation(wk1gt_4,0,112). coords(wk1gt_4,44,187,112).  visible(wk1gt_4,112). happensAt(walking(wk1gt_4),112).
orientation(wk1gt_0,131,113). coords(wk1gt_0,204,124,113).  visible(wk1gt_0,113). happensAt(inactive(wk1gt_0),113). orientation(wk1gt_4,0,113). coords(wk1gt_4,44,187,113).  visible(wk1gt_4,113). happensAt(walking(wk1gt_4),113).
orientation(wk1gt_0,131,114). coords(wk1gt_0,205,124,114).  visible(wk1gt_0,114). happensAt(inactive(wk1gt_0),114). orientation(wk1gt_4,0,114). coords(wk1gt_4,44,187,114).  visible(wk1gt_4,114). happensAt(walking(wk1gt_4),114).
orientation(wk1gt_0,131,115). coords(wk1gt_0,205,125,115).  visible(wk1gt_0,115). happensAt(inactive(wk1gt_0),115). orientation(wk1gt_4,0,115). coords(wk1gt_4,44,187,115).  visible(wk1gt_4,115). happensAt(walking(wk1gt_4),115).
orientation(wk1gt_0,131,116). coords(wk1gt_0,206,125,116).  visible(wk1gt_0,116). happensAt(inactive(wk1gt_0),116). orientation(wk1gt_4,0,116). coords(wk1gt_4,44,187,116).  visible(wk1gt_4,116). happensAt(walking(wk1gt_4),116).
orientation(wk1gt_0,131,117). coords(wk1gt_0,207,125,117).  visible(wk1gt_0,117). happensAt(inactive(wk1gt_0),117). orientation(wk1gt_4,0,117). coords(wk1gt_4,44,187,117).  visible(wk1gt_4,117). happensAt(walking(wk1gt_4),117).
orientation(wk1gt_0,131,118). coords(wk1gt_0,207,126,118).  visible(wk1gt_0,118). happensAt(active(wk1gt_0),118). orientation(wk1gt_4,0,118). coords(wk1gt_4,44,187,118).  visible(wk1gt_4,118). happensAt(walking(wk1gt_4),118).
orientation(wk1gt_0,131,119). coords(wk1gt_0,206,126,119).  visible(wk1gt_0,119). happensAt(active(wk1gt_0),119). orientation(wk1gt_4,0,119). coords(wk1gt_4,44,187,119).  visible(wk1gt_4,119). happensAt(walking(wk1gt_4),119).
orientation(wk1gt_0,131,120). coords(wk1gt_0,208,128,120).  visible(wk1gt_0,120). happensAt(active(wk1gt_0),120). orientation(wk1gt_4,0,120). coords(wk1gt_4,44,187,120).  visible(wk1gt_4,120). happensAt(walking(wk1gt_4),120).
orientation(wk1gt_0,131,121). coords(wk1gt_0,210,129,121).  visible(wk1gt_0,121). happensAt(active(wk1gt_0),121). orientation(wk1gt_4,0,121). coords(wk1gt_4,45,186,121).  visible(wk1gt_4,121). happensAt(walking(wk1gt_4),121).
orientation(wk1gt_0,131,122). coords(wk1gt_0,210,131,122).  visible(wk1gt_0,122). happensAt(active(wk1gt_0),122). orientation(wk1gt_4,0,122). coords(wk1gt_4,45,186,122).  visible(wk1gt_4,122). happensAt(walking(wk1gt_4),122).
orientation(wk1gt_0,131,123). coords(wk1gt_0,212,133,123).  visible(wk1gt_0,123). happensAt(active(wk1gt_0),123). orientation(wk1gt_4,0,123). coords(wk1gt_4,45,186,123).  visible(wk1gt_4,123). happensAt(walking(wk1gt_4),123).
#show holdsAt/2.