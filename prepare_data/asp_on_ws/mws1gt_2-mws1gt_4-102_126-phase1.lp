
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

	orientation(mws1gt_2,0,102). coords(mws1gt_2,34,191,102).  visible(mws1gt_2,102). happensAt(walking(mws1gt_2),102). orientation(mws1gt_4,0,102). coords(mws1gt_4,23,162,102).  happensAt(appear(mws1gt_4),102). happensAt(walking(mws1gt_4),102).
orientation(mws1gt_2,167,103). coords(mws1gt_2,34,192,103).  visible(mws1gt_2,103). happensAt(walking(mws1gt_2),103). orientation(mws1gt_4,0,103). coords(mws1gt_4,26,162,103).  visible(mws1gt_4,103). happensAt(walking(mws1gt_4),103).
orientation(mws1gt_2,167,104). coords(mws1gt_2,34,193,104).  visible(mws1gt_2,104). happensAt(walking(mws1gt_2),104). orientation(mws1gt_4,0,104). coords(mws1gt_4,26,162,104).  visible(mws1gt_4,104). happensAt(walking(mws1gt_4),104).
orientation(mws1gt_2,167,105). coords(mws1gt_2,34,194,105).  visible(mws1gt_2,105). happensAt(walking(mws1gt_2),105). orientation(mws1gt_4,0,105). coords(mws1gt_4,26,162,105).  visible(mws1gt_4,105). happensAt(walking(mws1gt_4),105).
orientation(mws1gt_2,164,106). coords(mws1gt_2,34,195,106).  visible(mws1gt_2,106). happensAt(walking(mws1gt_2),106). orientation(mws1gt_4,0,106). coords(mws1gt_4,26,160,106).  visible(mws1gt_4,106). happensAt(walking(mws1gt_4),106).
orientation(mws1gt_2,164,107). coords(mws1gt_2,34,196,107).  visible(mws1gt_2,107). happensAt(walking(mws1gt_2),107). orientation(mws1gt_4,0,107). coords(mws1gt_4,26,159,107).  visible(mws1gt_4,107). happensAt(walking(mws1gt_4),107).
orientation(mws1gt_2,164,108). coords(mws1gt_2,34,197,108).  visible(mws1gt_2,108). happensAt(walking(mws1gt_2),108). orientation(mws1gt_4,0,108). coords(mws1gt_4,26,159,108).  visible(mws1gt_4,108). happensAt(walking(mws1gt_4),108).
orientation(mws1gt_2,164,109). coords(mws1gt_2,34,197,109).  visible(mws1gt_2,109). happensAt(walking(mws1gt_2),109). orientation(mws1gt_4,0,109). coords(mws1gt_4,26,157,109).  visible(mws1gt_4,109). happensAt(walking(mws1gt_4),109).
orientation(mws1gt_2,164,110). coords(mws1gt_2,34,197,110).  visible(mws1gt_2,110). happensAt(walking(mws1gt_2),110). orientation(mws1gt_4,0,110). coords(mws1gt_4,26,157,110).  visible(mws1gt_4,110). happensAt(walking(mws1gt_4),110).
orientation(mws1gt_2,164,111). coords(mws1gt_2,34,198,111).  visible(mws1gt_2,111). happensAt(walking(mws1gt_2),111). orientation(mws1gt_4,0,111). coords(mws1gt_4,26,156,111).  visible(mws1gt_4,111). happensAt(walking(mws1gt_4),111).
orientation(mws1gt_2,164,112). coords(mws1gt_2,34,198,112).  visible(mws1gt_2,112). happensAt(walking(mws1gt_2),112). orientation(mws1gt_4,0,112). coords(mws1gt_4,26,155,112).  visible(mws1gt_4,112). happensAt(walking(mws1gt_4),112).
orientation(mws1gt_2,164,113). coords(mws1gt_2,34,200,113).  visible(mws1gt_2,113). happensAt(walking(mws1gt_2),113). orientation(mws1gt_4,0,113). coords(mws1gt_4,26,154,113).  visible(mws1gt_4,113). happensAt(walking(mws1gt_4),113).
orientation(mws1gt_2,164,114). coords(mws1gt_2,34,200,114).  visible(mws1gt_2,114). happensAt(walking(mws1gt_2),114). orientation(mws1gt_4,0,114). coords(mws1gt_4,26,153,114).  visible(mws1gt_4,114). happensAt(walking(mws1gt_4),114).
orientation(mws1gt_2,164,115). coords(mws1gt_2,34,201,115).  visible(mws1gt_2,115). happensAt(walking(mws1gt_2),115). orientation(mws1gt_4,0,115). coords(mws1gt_4,26,152,115).  visible(mws1gt_4,115). happensAt(walking(mws1gt_4),115).
orientation(mws1gt_2,164,116). coords(mws1gt_2,34,202,116).  visible(mws1gt_2,116). happensAt(walking(mws1gt_2),116). orientation(mws1gt_4,0,116). coords(mws1gt_4,26,151,116).  visible(mws1gt_4,116). happensAt(walking(mws1gt_4),116).
orientation(mws1gt_2,164,117). coords(mws1gt_2,34,203,117).  visible(mws1gt_2,117). happensAt(walking(mws1gt_2),117). orientation(mws1gt_4,0,117). coords(mws1gt_4,26,151,117).  visible(mws1gt_4,117). happensAt(walking(mws1gt_4),117).
orientation(mws1gt_2,164,118). coords(mws1gt_2,34,203,118).  visible(mws1gt_2,118). happensAt(walking(mws1gt_2),118). orientation(mws1gt_4,0,118). coords(mws1gt_4,27,150,118).  visible(mws1gt_4,118). happensAt(walking(mws1gt_4),118).
orientation(mws1gt_2,164,119). coords(mws1gt_2,34,204,119).  visible(mws1gt_2,119). happensAt(walking(mws1gt_2),119). orientation(mws1gt_4,0,119). coords(mws1gt_4,27,149,119).  visible(mws1gt_4,119). happensAt(walking(mws1gt_4),119).
orientation(mws1gt_2,164,120). coords(mws1gt_2,34,205,120).  visible(mws1gt_2,120). happensAt(walking(mws1gt_2),120). orientation(mws1gt_4,0,120). coords(mws1gt_4,27,148,120).  visible(mws1gt_4,120). happensAt(walking(mws1gt_4),120).
orientation(mws1gt_2,164,121). coords(mws1gt_2,34,205,121).  visible(mws1gt_2,121). happensAt(walking(mws1gt_2),121). orientation(mws1gt_4,0,121). coords(mws1gt_4,27,148,121).  visible(mws1gt_4,121). happensAt(walking(mws1gt_4),121).
orientation(mws1gt_2,164,122). coords(mws1gt_2,33,205,122).  visible(mws1gt_2,122). happensAt(walking(mws1gt_2),122). orientation(mws1gt_4,0,122). coords(mws1gt_4,28,147,122).  visible(mws1gt_4,122). happensAt(walking(mws1gt_4),122).
orientation(mws1gt_2,164,123). coords(mws1gt_2,33,205,123).  visible(mws1gt_2,123). happensAt(walking(mws1gt_2),123). orientation(mws1gt_4,0,123). coords(mws1gt_4,27,146,123).  visible(mws1gt_4,123). happensAt(walking(mws1gt_4),123).
orientation(mws1gt_2,164,124). coords(mws1gt_2,33,206,124).  visible(mws1gt_2,124). happensAt(walking(mws1gt_2),124). orientation(mws1gt_4,0,124). coords(mws1gt_4,27,145,124).  visible(mws1gt_4,124). happensAt(walking(mws1gt_4),124).
orientation(mws1gt_2,164,125). coords(mws1gt_2,32,206,125).  visible(mws1gt_2,125). happensAt(walking(mws1gt_2),125). orientation(mws1gt_4,0,125). coords(mws1gt_4,27,144,125).  visible(mws1gt_4,125). happensAt(walking(mws1gt_4),125).
orientation(mws1gt_2,164,126). coords(mws1gt_2,32,206,126).  visible(mws1gt_2,126). happensAt(walking(mws1gt_2),126). orientation(mws1gt_4,0,126). coords(mws1gt_4,27,144,126).  visible(mws1gt_4,126). happensAt(walking(mws1gt_4),126).
#show holdsAt/2.