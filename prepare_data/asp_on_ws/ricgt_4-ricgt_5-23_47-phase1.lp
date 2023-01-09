
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

	orientation(ricgt_4,172,23). coords(ricgt_4,32,118,23).  visible(ricgt_4,23). happensAt(walking(ricgt_4),23). orientation(ricgt_5,0,23). coords(ricgt_5,30,114,23).  visible(ricgt_5,23). happensAt(walking(ricgt_5),23).
orientation(ricgt_4,172,24). coords(ricgt_4,31,118,24).  visible(ricgt_4,24). happensAt(walking(ricgt_4),24). orientation(ricgt_5,0,24). coords(ricgt_5,30,114,24).  visible(ricgt_5,24). happensAt(walking(ricgt_5),24).
orientation(ricgt_4,172,25). coords(ricgt_4,31,118,25).  visible(ricgt_4,25). happensAt(walking(ricgt_4),25). orientation(ricgt_5,0,25). coords(ricgt_5,30,114,25).  visible(ricgt_5,25). happensAt(walking(ricgt_5),25).
orientation(ricgt_4,172,26). coords(ricgt_4,31,118,26).  visible(ricgt_4,26). happensAt(walking(ricgt_4),26). orientation(ricgt_5,0,26). coords(ricgt_5,30,114,26).  visible(ricgt_5,26). happensAt(walking(ricgt_5),26).
orientation(ricgt_4,172,27). coords(ricgt_4,31,118,27).  visible(ricgt_4,27). happensAt(walking(ricgt_4),27). orientation(ricgt_5,0,27). coords(ricgt_5,30,114,27).  visible(ricgt_5,27). happensAt(walking(ricgt_5),27).
orientation(ricgt_4,172,28). coords(ricgt_4,31,118,28).  visible(ricgt_4,28). happensAt(walking(ricgt_4),28). orientation(ricgt_5,0,28). coords(ricgt_5,30,114,28).  visible(ricgt_5,28). happensAt(walking(ricgt_5),28).
orientation(ricgt_4,172,29). coords(ricgt_4,31,118,29).  visible(ricgt_4,29). happensAt(walking(ricgt_4),29). orientation(ricgt_5,0,29). coords(ricgt_5,30,114,29).  visible(ricgt_5,29). happensAt(walking(ricgt_5),29).
orientation(ricgt_4,172,30). coords(ricgt_4,31,118,30).  visible(ricgt_4,30). happensAt(walking(ricgt_4),30). orientation(ricgt_5,0,30). coords(ricgt_5,30,114,30).  visible(ricgt_5,30). happensAt(walking(ricgt_5),30).
orientation(ricgt_4,172,31). coords(ricgt_4,31,120,31).  visible(ricgt_4,31). happensAt(walking(ricgt_4),31). orientation(ricgt_5,0,31). coords(ricgt_5,30,114,31).  visible(ricgt_5,31). happensAt(walking(ricgt_5),31).
orientation(ricgt_4,172,32). coords(ricgt_4,31,120,32).  visible(ricgt_4,32). happensAt(walking(ricgt_4),32). orientation(ricgt_5,0,32). coords(ricgt_5,30,115,32).  visible(ricgt_5,32). happensAt(walking(ricgt_5),32).
orientation(ricgt_4,172,33). coords(ricgt_4,31,121,33).  visible(ricgt_4,33). happensAt(walking(ricgt_4),33). orientation(ricgt_5,0,33). coords(ricgt_5,30,116,33).  visible(ricgt_5,33). happensAt(walking(ricgt_5),33).
orientation(ricgt_4,172,34). coords(ricgt_4,31,121,34).  visible(ricgt_4,34). happensAt(walking(ricgt_4),34). orientation(ricgt_5,0,34). coords(ricgt_5,30,116,34).  visible(ricgt_5,34). happensAt(walking(ricgt_5),34).
orientation(ricgt_4,172,35). coords(ricgt_4,31,122,35).  visible(ricgt_4,35). happensAt(walking(ricgt_4),35). orientation(ricgt_5,0,35). coords(ricgt_5,30,117,35).  visible(ricgt_5,35). happensAt(walking(ricgt_5),35).
orientation(ricgt_4,172,36). coords(ricgt_4,31,122,36).  visible(ricgt_4,36). happensAt(walking(ricgt_4),36). orientation(ricgt_5,0,36). coords(ricgt_5,30,117,36).  visible(ricgt_5,36). happensAt(walking(ricgt_5),36).
orientation(ricgt_4,172,37). coords(ricgt_4,31,122,37).  visible(ricgt_4,37). happensAt(walking(ricgt_4),37). orientation(ricgt_5,0,37). coords(ricgt_5,30,117,37).  visible(ricgt_5,37). happensAt(walking(ricgt_5),37).
orientation(ricgt_4,172,38). coords(ricgt_4,31,123,38).  visible(ricgt_4,38). happensAt(walking(ricgt_4),38). orientation(ricgt_5,0,38). coords(ricgt_5,30,117,38).  visible(ricgt_5,38). happensAt(walking(ricgt_5),38).
orientation(ricgt_4,172,39). coords(ricgt_4,31,123,39).  visible(ricgt_4,39). happensAt(walking(ricgt_4),39). orientation(ricgt_5,0,39). coords(ricgt_5,30,117,39).  visible(ricgt_5,39). happensAt(walking(ricgt_5),39).
orientation(ricgt_4,172,40). coords(ricgt_4,31,123,40).  visible(ricgt_4,40). happensAt(walking(ricgt_4),40). orientation(ricgt_5,0,40). coords(ricgt_5,30,117,40).  visible(ricgt_5,40). happensAt(walking(ricgt_5),40).
orientation(ricgt_4,172,41). coords(ricgt_4,31,123,41).  visible(ricgt_4,41). happensAt(walking(ricgt_4),41). orientation(ricgt_5,0,41). coords(ricgt_5,30,117,41).  visible(ricgt_5,41). happensAt(walking(ricgt_5),41).
orientation(ricgt_4,172,42). coords(ricgt_4,31,123,42).  visible(ricgt_4,42). happensAt(walking(ricgt_4),42). orientation(ricgt_5,0,42). coords(ricgt_5,30,117,42).  visible(ricgt_5,42). happensAt(walking(ricgt_5),42).
orientation(ricgt_4,172,43). coords(ricgt_4,30,124,43).  visible(ricgt_4,43). happensAt(walking(ricgt_4),43). orientation(ricgt_5,0,43). coords(ricgt_5,30,117,43).  visible(ricgt_5,43). happensAt(walking(ricgt_5),43).
orientation(ricgt_4,172,44). coords(ricgt_4,30,125,44).  visible(ricgt_4,44). happensAt(walking(ricgt_4),44). orientation(ricgt_5,0,44). coords(ricgt_5,30,117,44).  visible(ricgt_5,44). happensAt(walking(ricgt_5),44).
orientation(ricgt_4,172,45). coords(ricgt_4,30,125,45).  visible(ricgt_4,45). happensAt(walking(ricgt_4),45). orientation(ricgt_5,0,45). coords(ricgt_5,30,119,45).  visible(ricgt_5,45). happensAt(walking(ricgt_5),45).
orientation(ricgt_4,172,46). coords(ricgt_4,30,125,46).  visible(ricgt_4,46). happensAt(walking(ricgt_4),46). orientation(ricgt_5,0,46). coords(ricgt_5,30,119,46).  visible(ricgt_5,46). happensAt(walking(ricgt_5),46).
orientation(ricgt_4,172,47). coords(ricgt_4,30,125,47).  visible(ricgt_4,47). happensAt(walking(ricgt_4),47). orientation(ricgt_5,0,47). coords(ricgt_5,30,119,47).  visible(ricgt_5,47). happensAt(walking(ricgt_5),47).
#show holdsAt/2.