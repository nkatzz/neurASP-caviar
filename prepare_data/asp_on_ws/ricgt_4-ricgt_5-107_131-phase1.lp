
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

	orientation(ricgt_4,172,107). coords(ricgt_4,26,134,107).  visible(ricgt_4,107). happensAt(walking(ricgt_4),107). orientation(ricgt_5,0,107). coords(ricgt_5,27,127,107).  visible(ricgt_5,107). happensAt(walking(ricgt_5),107).
orientation(ricgt_4,172,108). coords(ricgt_4,26,134,108).  visible(ricgt_4,108). happensAt(walking(ricgt_4),108). orientation(ricgt_5,0,108). coords(ricgt_5,27,127,108).  visible(ricgt_5,108). happensAt(walking(ricgt_5),108).
orientation(ricgt_4,172,109). coords(ricgt_4,26,134,109).  visible(ricgt_4,109). happensAt(walking(ricgt_4),109). orientation(ricgt_5,0,109). coords(ricgt_5,27,127,109).  visible(ricgt_5,109). happensAt(walking(ricgt_5),109).
orientation(ricgt_4,172,110). coords(ricgt_4,26,134,110).  visible(ricgt_4,110). happensAt(walking(ricgt_4),110). orientation(ricgt_5,0,110). coords(ricgt_5,27,127,110).  visible(ricgt_5,110). happensAt(walking(ricgt_5),110).
orientation(ricgt_4,172,111). coords(ricgt_4,26,134,111).  visible(ricgt_4,111). happensAt(walking(ricgt_4),111). orientation(ricgt_5,0,111). coords(ricgt_5,27,127,111).  visible(ricgt_5,111). happensAt(walking(ricgt_5),111).
orientation(ricgt_4,172,112). coords(ricgt_4,26,134,112).  visible(ricgt_4,112). happensAt(walking(ricgt_4),112). orientation(ricgt_5,0,112). coords(ricgt_5,27,127,112).  visible(ricgt_5,112). happensAt(walking(ricgt_5),112).
orientation(ricgt_4,172,113). coords(ricgt_4,26,134,113).  visible(ricgt_4,113). happensAt(walking(ricgt_4),113). orientation(ricgt_5,0,113). coords(ricgt_5,27,127,113).  visible(ricgt_5,113). happensAt(walking(ricgt_5),113).
orientation(ricgt_4,172,114). coords(ricgt_4,26,134,114).  visible(ricgt_4,114). happensAt(walking(ricgt_4),114). orientation(ricgt_5,0,114). coords(ricgt_5,27,127,114).  visible(ricgt_5,114). happensAt(walking(ricgt_5),114).
orientation(ricgt_4,172,115). coords(ricgt_4,26,134,115).  visible(ricgt_4,115). happensAt(walking(ricgt_4),115). orientation(ricgt_5,0,115). coords(ricgt_5,27,127,115).  visible(ricgt_5,115). happensAt(walking(ricgt_5),115).
orientation(ricgt_4,172,116). coords(ricgt_4,26,134,116).  visible(ricgt_4,116). happensAt(walking(ricgt_4),116). orientation(ricgt_5,0,116). coords(ricgt_5,27,127,116).  visible(ricgt_5,116). happensAt(walking(ricgt_5),116).
orientation(ricgt_4,172,117). coords(ricgt_4,26,134,117).  visible(ricgt_4,117). happensAt(walking(ricgt_4),117). orientation(ricgt_5,0,117). coords(ricgt_5,27,127,117).  visible(ricgt_5,117). happensAt(walking(ricgt_5),117).
orientation(ricgt_4,172,118). coords(ricgt_4,26,134,118).  visible(ricgt_4,118). happensAt(walking(ricgt_4),118). orientation(ricgt_5,0,118). coords(ricgt_5,27,127,118).  visible(ricgt_5,118). happensAt(walking(ricgt_5),118).
orientation(ricgt_4,172,119). coords(ricgt_4,26,134,119).  visible(ricgt_4,119). happensAt(walking(ricgt_4),119). orientation(ricgt_5,0,119). coords(ricgt_5,27,127,119).  visible(ricgt_5,119). happensAt(walking(ricgt_5),119).
orientation(ricgt_4,172,120). coords(ricgt_4,26,134,120).  visible(ricgt_4,120). happensAt(walking(ricgt_4),120). orientation(ricgt_5,0,120). coords(ricgt_5,27,127,120).  visible(ricgt_5,120). happensAt(walking(ricgt_5),120).
orientation(ricgt_4,172,121). coords(ricgt_4,26,134,121).  visible(ricgt_4,121). happensAt(walking(ricgt_4),121). orientation(ricgt_5,0,121). coords(ricgt_5,27,127,121).  visible(ricgt_5,121). happensAt(walking(ricgt_5),121).
orientation(ricgt_4,172,122). coords(ricgt_4,26,134,122).  visible(ricgt_4,122). happensAt(walking(ricgt_4),122). orientation(ricgt_5,0,122). coords(ricgt_5,27,127,122).  visible(ricgt_5,122). happensAt(walking(ricgt_5),122).
orientation(ricgt_4,172,123). coords(ricgt_4,26,134,123).  visible(ricgt_4,123). happensAt(walking(ricgt_4),123). orientation(ricgt_5,0,123). coords(ricgt_5,27,127,123).  visible(ricgt_5,123). happensAt(walking(ricgt_5),123).
orientation(ricgt_4,172,124). coords(ricgt_4,26,134,124).  visible(ricgt_4,124). happensAt(walking(ricgt_4),124). orientation(ricgt_5,0,124). coords(ricgt_5,27,127,124).  visible(ricgt_5,124). happensAt(walking(ricgt_5),124).
orientation(ricgt_4,172,125). coords(ricgt_4,26,134,125).  visible(ricgt_4,125). happensAt(walking(ricgt_4),125). orientation(ricgt_5,0,125). coords(ricgt_5,27,127,125).  visible(ricgt_5,125). happensAt(walking(ricgt_5),125).
orientation(ricgt_4,172,126). coords(ricgt_4,26,134,126).  visible(ricgt_4,126). happensAt(walking(ricgt_4),126). orientation(ricgt_5,0,126). coords(ricgt_5,27,127,126).  visible(ricgt_5,126). happensAt(walking(ricgt_5),126).
orientation(ricgt_4,172,127). coords(ricgt_4,26,134,127).  visible(ricgt_4,127). happensAt(walking(ricgt_4),127). orientation(ricgt_5,0,127). coords(ricgt_5,27,127,127).  visible(ricgt_5,127). happensAt(walking(ricgt_5),127).
orientation(ricgt_4,172,128). coords(ricgt_4,26,134,128).  visible(ricgt_4,128). happensAt(walking(ricgt_4),128). orientation(ricgt_5,0,128). coords(ricgt_5,27,127,128).  visible(ricgt_5,128). happensAt(walking(ricgt_5),128).
orientation(ricgt_4,172,129). coords(ricgt_4,26,134,129).  visible(ricgt_4,129). happensAt(walking(ricgt_4),129). orientation(ricgt_5,0,129). coords(ricgt_5,27,127,129).  visible(ricgt_5,129). happensAt(walking(ricgt_5),129).
orientation(ricgt_4,172,130). coords(ricgt_4,26,134,130).  visible(ricgt_4,130). happensAt(walking(ricgt_4),130). orientation(ricgt_5,0,130). coords(ricgt_5,27,127,130).  visible(ricgt_5,130). happensAt(walking(ricgt_5),130).
orientation(ricgt_4,172,131). coords(ricgt_4,26,134,131).  visible(ricgt_4,131). happensAt(walking(ricgt_4),131). orientation(ricgt_5,0,131). coords(ricgt_5,27,127,131).  visible(ricgt_5,131). happensAt(walking(ricgt_5),131).
#show holdsAt/2.