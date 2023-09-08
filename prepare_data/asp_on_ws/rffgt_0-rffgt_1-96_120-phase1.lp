
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

	orientation(rffgt_0,135,96). coords(rffgt_0,139,46,96).  visible(rffgt_0,96). happensAt(walking(rffgt_0),96). orientation(rffgt_1,146,96). coords(rffgt_1,150,106,96).  visible(rffgt_1,96). happensAt(active(rffgt_1),96).
orientation(rffgt_0,135,97). coords(rffgt_0,139,46,97).  visible(rffgt_0,97). happensAt(walking(rffgt_0),97). orientation(rffgt_1,146,97). coords(rffgt_1,150,106,97).  visible(rffgt_1,97). happensAt(active(rffgt_1),97).
orientation(rffgt_0,135,98). coords(rffgt_0,139,44,98).  visible(rffgt_0,98). happensAt(walking(rffgt_0),98). orientation(rffgt_1,146,98). coords(rffgt_1,150,107,98).  visible(rffgt_1,98). happensAt(active(rffgt_1),98).
orientation(rffgt_0,135,99). coords(rffgt_0,139,44,99).  visible(rffgt_0,99). happensAt(walking(rffgt_0),99). orientation(rffgt_1,146,99). coords(rffgt_1,151,107,99).  visible(rffgt_1,99). happensAt(active(rffgt_1),99).
orientation(rffgt_0,135,100). coords(rffgt_0,139,43,100).  visible(rffgt_0,100). happensAt(walking(rffgt_0),100). orientation(rffgt_1,144,100). coords(rffgt_1,151,108,100).  visible(rffgt_1,100). happensAt(active(rffgt_1),100).
orientation(rffgt_0,135,101). coords(rffgt_0,139,43,101).  visible(rffgt_0,101). happensAt(walking(rffgt_0),101). orientation(rffgt_1,147,101). coords(rffgt_1,152,108,101).  visible(rffgt_1,101). happensAt(active(rffgt_1),101).
orientation(rffgt_0,135,102). coords(rffgt_0,139,42,102).  visible(rffgt_0,102). happensAt(walking(rffgt_0),102). orientation(rffgt_1,147,102). coords(rffgt_1,152,108,102).  visible(rffgt_1,102). happensAt(active(rffgt_1),102).
orientation(rffgt_0,135,103). coords(rffgt_0,139,42,103).  visible(rffgt_0,103). happensAt(walking(rffgt_0),103). orientation(rffgt_1,145,103). coords(rffgt_1,152,108,103).  visible(rffgt_1,103). happensAt(active(rffgt_1),103).
orientation(rffgt_0,135,104). coords(rffgt_0,139,42,104).  visible(rffgt_0,104). happensAt(walking(rffgt_0),104). orientation(rffgt_1,145,104). coords(rffgt_1,152,108,104).  visible(rffgt_1,104). happensAt(active(rffgt_1),104).
orientation(rffgt_0,135,105). coords(rffgt_0,140,42,105).  visible(rffgt_0,105). happensAt(walking(rffgt_0),105). orientation(rffgt_1,146,105). coords(rffgt_1,151,108,105).  visible(rffgt_1,105). happensAt(active(rffgt_1),105).
orientation(rffgt_0,135,106). coords(rffgt_0,140,42,106).  visible(rffgt_0,106). happensAt(walking(rffgt_0),106). orientation(rffgt_1,146,106). coords(rffgt_1,151,108,106).  visible(rffgt_1,106). happensAt(active(rffgt_1),106).
orientation(rffgt_0,135,107). coords(rffgt_0,141,42,107).  visible(rffgt_0,107). happensAt(walking(rffgt_0),107). orientation(rffgt_1,146,107). coords(rffgt_1,152,109,107).  visible(rffgt_1,107). happensAt(inactive(rffgt_1),107).
orientation(rffgt_0,135,108). coords(rffgt_0,142,42,108).  visible(rffgt_0,108). happensAt(walking(rffgt_0),108). orientation(rffgt_1,146,108). coords(rffgt_1,152,109,108).  visible(rffgt_1,108). happensAt(inactive(rffgt_1),108).
orientation(rffgt_0,135,109). coords(rffgt_0,143,42,109).  visible(rffgt_0,109). happensAt(walking(rffgt_0),109). orientation(rffgt_1,146,109). coords(rffgt_1,152,110,109).  visible(rffgt_1,109). happensAt(inactive(rffgt_1),109).
orientation(rffgt_0,135,110). coords(rffgt_0,144,42,110).  visible(rffgt_0,110). happensAt(walking(rffgt_0),110). orientation(rffgt_1,146,110). coords(rffgt_1,152,109,110).  visible(rffgt_1,110). happensAt(inactive(rffgt_1),110).
orientation(rffgt_0,135,111). coords(rffgt_0,144,41,111).  visible(rffgt_0,111). happensAt(walking(rffgt_0),111). orientation(rffgt_1,146,111). coords(rffgt_1,152,110,111).  visible(rffgt_1,111). happensAt(inactive(rffgt_1),111).
orientation(rffgt_0,135,112). coords(rffgt_0,144,41,112).  visible(rffgt_0,112). happensAt(walking(rffgt_0),112). orientation(rffgt_1,146,112). coords(rffgt_1,152,110,112).  visible(rffgt_1,112). happensAt(inactive(rffgt_1),112).
orientation(rffgt_0,135,113). coords(rffgt_0,144,41,113).  visible(rffgt_0,113). happensAt(walking(rffgt_0),113). orientation(rffgt_1,146,113). coords(rffgt_1,152,110,113).  visible(rffgt_1,113). happensAt(inactive(rffgt_1),113).
orientation(rffgt_0,135,114). coords(rffgt_0,145,41,114).  visible(rffgt_0,114). happensAt(walking(rffgt_0),114). orientation(rffgt_1,146,114). coords(rffgt_1,152,110,114).  visible(rffgt_1,114). happensAt(inactive(rffgt_1),114).
orientation(rffgt_0,135,115). coords(rffgt_0,145,41,115).  visible(rffgt_0,115). happensAt(walking(rffgt_0),115). orientation(rffgt_1,146,115). coords(rffgt_1,151,109,115).  visible(rffgt_1,115). happensAt(inactive(rffgt_1),115).
orientation(rffgt_0,135,116). coords(rffgt_0,145,40,116).  visible(rffgt_0,116). happensAt(walking(rffgt_0),116). orientation(rffgt_1,146,116). coords(rffgt_1,151,109,116).  visible(rffgt_1,116). happensAt(inactive(rffgt_1),116).
orientation(rffgt_0,135,117). coords(rffgt_0,146,39,117).  visible(rffgt_0,117). happensAt(walking(rffgt_0),117). orientation(rffgt_1,146,117). coords(rffgt_1,151,109,117).  visible(rffgt_1,117). happensAt(inactive(rffgt_1),117).
orientation(rffgt_0,135,118). coords(rffgt_0,146,39,118).  visible(rffgt_0,118). happensAt(walking(rffgt_0),118). orientation(rffgt_1,146,118). coords(rffgt_1,152,110,118).  visible(rffgt_1,118). happensAt(inactive(rffgt_1),118).
orientation(rffgt_0,135,119). coords(rffgt_0,146,39,119).  visible(rffgt_0,119). happensAt(walking(rffgt_0),119). orientation(rffgt_1,146,119). coords(rffgt_1,152,110,119).  visible(rffgt_1,119). happensAt(inactive(rffgt_1),119).
orientation(rffgt_0,135,120). coords(rffgt_0,148,39,120).  visible(rffgt_0,120). happensAt(walking(rffgt_0),120). orientation(rffgt_1,146,120). coords(rffgt_1,152,110,120).  visible(rffgt_1,120). happensAt(inactive(rffgt_1),120).
#show holdsAt/2.