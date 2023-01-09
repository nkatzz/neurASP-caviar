
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

	orientation(rffgt_0,144,24). coords(rffgt_0,143,69,24).  visible(rffgt_0,24). happensAt(active(rffgt_0),24). orientation(rffgt_1,141,24). coords(rffgt_1,166,105,24).  visible(rffgt_1,24). happensAt(active(rffgt_1),24).
orientation(rffgt_0,144,25). coords(rffgt_0,143,69,25).  visible(rffgt_0,25). happensAt(active(rffgt_0),25). orientation(rffgt_1,138,25). coords(rffgt_1,166,105,25).  visible(rffgt_1,25). happensAt(active(rffgt_1),25).
orientation(rffgt_0,146,26). coords(rffgt_0,143,68,26).  visible(rffgt_0,26). happensAt(active(rffgt_0),26). orientation(rffgt_1,138,26). coords(rffgt_1,166,105,26).  visible(rffgt_1,26). happensAt(active(rffgt_1),26).
orientation(rffgt_0,146,27). coords(rffgt_0,143,68,27).  visible(rffgt_0,27). happensAt(active(rffgt_0),27). orientation(rffgt_1,138,27). coords(rffgt_1,165,105,27).  visible(rffgt_1,27). happensAt(active(rffgt_1),27).
orientation(rffgt_0,146,28). coords(rffgt_0,143,68,28).  visible(rffgt_0,28). happensAt(active(rffgt_0),28). orientation(rffgt_1,138,28). coords(rffgt_1,165,104,28).  visible(rffgt_1,28). happensAt(active(rffgt_1),28).
orientation(rffgt_0,146,29). coords(rffgt_0,143,68,29).  visible(rffgt_0,29). happensAt(active(rffgt_0),29). orientation(rffgt_1,138,29). coords(rffgt_1,165,104,29).  visible(rffgt_1,29). happensAt(active(rffgt_1),29).
orientation(rffgt_0,146,30). coords(rffgt_0,142,68,30).  visible(rffgt_0,30). happensAt(active(rffgt_0),30). orientation(rffgt_1,138,30). coords(rffgt_1,164,103,30).  visible(rffgt_1,30). happensAt(active(rffgt_1),30).
orientation(rffgt_0,146,31). coords(rffgt_0,142,68,31).  visible(rffgt_0,31). happensAt(active(rffgt_0),31). orientation(rffgt_1,138,31). coords(rffgt_1,164,104,31).  visible(rffgt_1,31). happensAt(active(rffgt_1),31).
orientation(rffgt_0,148,32). coords(rffgt_0,142,68,32).  visible(rffgt_0,32). happensAt(active(rffgt_0),32). orientation(rffgt_1,141,32). coords(rffgt_1,163,104,32).  visible(rffgt_1,32). happensAt(active(rffgt_1),32).
orientation(rffgt_0,148,33). coords(rffgt_0,141,68,33).  visible(rffgt_0,33). happensAt(active(rffgt_0),33). orientation(rffgt_1,146,33). coords(rffgt_1,163,104,33).  visible(rffgt_1,33). happensAt(active(rffgt_1),33).
orientation(rffgt_0,148,34). coords(rffgt_0,140,68,34).  visible(rffgt_0,34). happensAt(active(rffgt_0),34). orientation(rffgt_1,151,34). coords(rffgt_1,162,105,34).  visible(rffgt_1,34). happensAt(active(rffgt_1),34).
orientation(rffgt_0,148,35). coords(rffgt_0,139,67,35).  visible(rffgt_0,35). happensAt(active(rffgt_0),35). orientation(rffgt_1,151,35). coords(rffgt_1,162,105,35).  visible(rffgt_1,35). happensAt(active(rffgt_1),35).
orientation(rffgt_0,148,36). coords(rffgt_0,138,67,36).  visible(rffgt_0,36). happensAt(active(rffgt_0),36). orientation(rffgt_1,151,36). coords(rffgt_1,161,105,36).  visible(rffgt_1,36). happensAt(active(rffgt_1),36).
orientation(rffgt_0,144,37). coords(rffgt_0,138,67,37).  visible(rffgt_0,37). happensAt(active(rffgt_0),37). orientation(rffgt_1,151,37). coords(rffgt_1,162,106,37).  visible(rffgt_1,37). happensAt(active(rffgt_1),37).
orientation(rffgt_0,144,38). coords(rffgt_0,138,67,38).  visible(rffgt_0,38). happensAt(active(rffgt_0),38). orientation(rffgt_1,151,38). coords(rffgt_1,161,106,38).  visible(rffgt_1,38). happensAt(active(rffgt_1),38).
orientation(rffgt_0,144,39). coords(rffgt_0,138,66,39).  visible(rffgt_0,39). happensAt(active(rffgt_0),39). orientation(rffgt_1,151,39). coords(rffgt_1,161,106,39).  visible(rffgt_1,39). happensAt(active(rffgt_1),39).
orientation(rffgt_0,144,40). coords(rffgt_0,138,66,40).  visible(rffgt_0,40). happensAt(active(rffgt_0),40). orientation(rffgt_1,154,40). coords(rffgt_1,161,106,40).  visible(rffgt_1,40). happensAt(active(rffgt_1),40).
orientation(rffgt_0,141,41). coords(rffgt_0,138,66,41).  visible(rffgt_0,41). happensAt(active(rffgt_0),41). orientation(rffgt_1,156,41). coords(rffgt_1,161,106,41).  visible(rffgt_1,41). happensAt(active(rffgt_1),41).
orientation(rffgt_0,141,42). coords(rffgt_0,138,66,42).  visible(rffgt_0,42). happensAt(active(rffgt_0),42). orientation(rffgt_1,156,42). coords(rffgt_1,161,106,42).  visible(rffgt_1,42). happensAt(active(rffgt_1),42).
orientation(rffgt_0,141,43). coords(rffgt_0,138,65,43).  visible(rffgt_0,43). happensAt(active(rffgt_0),43). orientation(rffgt_1,156,43). coords(rffgt_1,161,106,43).  visible(rffgt_1,43). happensAt(active(rffgt_1),43).
orientation(rffgt_0,141,44). coords(rffgt_0,138,65,44).  visible(rffgt_0,44). happensAt(active(rffgt_0),44). orientation(rffgt_1,158,44). coords(rffgt_1,161,106,44).  visible(rffgt_1,44). happensAt(active(rffgt_1),44).
orientation(rffgt_0,137,45). coords(rffgt_0,138,65,45).  visible(rffgt_0,45). happensAt(active(rffgt_0),45). orientation(rffgt_1,160,45). coords(rffgt_1,161,106,45).  visible(rffgt_1,45). happensAt(active(rffgt_1),45).
orientation(rffgt_0,137,46). coords(rffgt_0,139,65,46).  visible(rffgt_0,46). happensAt(active(rffgt_0),46). orientation(rffgt_1,162,46). coords(rffgt_1,161,106,46).  visible(rffgt_1,46). happensAt(active(rffgt_1),46).
orientation(rffgt_0,137,47). coords(rffgt_0,139,65,47).  visible(rffgt_0,47). happensAt(active(rffgt_0),47). orientation(rffgt_1,167,47). coords(rffgt_1,161,106,47).  visible(rffgt_1,47). happensAt(active(rffgt_1),47).
orientation(rffgt_0,137,48). coords(rffgt_0,138,63,48).  visible(rffgt_0,48). happensAt(active(rffgt_0),48). orientation(rffgt_1,173,48). coords(rffgt_1,161,106,48).  visible(rffgt_1,48). happensAt(active(rffgt_1),48).
#show holdsAt/2.